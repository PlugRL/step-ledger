"""Event streams, the ledger file format, and flushing open ledgers at exit.

A ledger file is an ``.npz`` (no pickles) holding one or more event streams,
concatenated column by column:

    meta        JSON string: format, side, reward_dtype, spaces, package version
    index       int64 (k,)    env index of each stream
    offsets     int64 (k+1,)  stream s is rows offsets[s]:offsets[s+1]
    kind        uint8         0 = reset, 1 = step
    h_obs       uint64        digest of the canonical observation
    h_act       uint64        digest of the canonical action (0 for resets)
    h_rew       uint64        raw bits of the reward cast to reward_dtype (0 for resets)
    terminated  bool
    truncated   bool          canonical: truncated and not terminated
    episode     int64         0-based episode counter (-1 before the first reset)
    step        int64         0 for a reset, k for the k-th step of the episode

With ``keep_values=True`` it also holds ``values_obs_{j}`` and
``values_act_{j}``: leaf ``j`` of the canonical value at every row (zeros where
there is no value, e.g. the action of a reset).
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import weakref
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from gymnasium import spaces

from stepledger._canon import Canon, reward_bits, reward_bits_batch, reward_dtype_of
from stepledger._version import __version__

FORMAT = 1
RESET = 0
STEP = 1
COLUMNS = ("kind", "h_obs", "h_act", "h_rew", "terminated", "truncated", "episode", "step")

Entry = tuple[int, "list[np.ndarray] | None"]  # (digest, kept leaves or None)


class Stream:
    """One sub-environment's events, held in compact arrays (27 bytes per event)."""

    def __init__(self, keep_values: bool) -> None:
        self.kind = array("B")
        self.h_obs = array("Q")
        self.h_act = array("Q")
        self.h_rew = array("Q")
        self.terminated = array("B")
        self.truncated = array("B")
        self.obs_values: list[list[np.ndarray]] | None = [] if keep_values else None
        self.act_values: list[list[np.ndarray] | None] | None = [] if keep_values else None

    def __len__(self) -> int:
        return len(self.kind)

    def add(self, kind: int, obs: Entry, act: Entry | None, rew: int, terminated: bool, truncated: bool) -> None:
        self.kind.append(kind)
        self.h_obs.append(obs[0])
        self.h_act.append(0 if act is None else act[0])
        self.h_rew.append(rew)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        if self.obs_values is not None:
            self.obs_values.append(obs[1])
            self.act_values.append(None if act is None else act[1])

    def columns(self) -> dict[str, np.ndarray]:
        kind = np.array(self.kind, dtype=np.uint8)
        is_reset = kind == RESET
        rows = np.arange(len(kind), dtype=np.int64)
        last_reset = np.maximum.accumulate(np.where(is_reset, rows, -1)) if len(kind) else rows
        return {
            "kind": kind,
            "h_obs": np.array(self.h_obs, dtype=np.uint64),
            "h_act": np.array(self.h_act, dtype=np.uint64),
            "h_rew": np.array(self.h_rew, dtype=np.uint64),
            "terminated": np.array(self.terminated, dtype=np.uint8).astype(bool),
            "truncated": np.array(self.truncated, dtype=np.uint8).astype(bool),
            "episode": np.cumsum(is_reset, dtype=np.int64) - 1,
            "step": rows - last_reset,
        }


class Recorder:
    """Canonicalizes values and appends events to one stream per sub-environment."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_dtype: Any,
        keep_values: bool,
        num_streams: int,
    ) -> None:
        self.obs = Canon(observation_space)
        self.act = Canon(action_space)
        self.reward_dtype = reward_dtype_of(reward_dtype)
        self.keep_values = bool(keep_values)
        self.streams = [Stream(self.keep_values) for _ in range(num_streams)]

    def _entry(self, canon: Canon, leaves: Sequence[np.ndarray]) -> Entry:
        kept = [np.array(leaf, copy=True) for leaf in leaves] if self.keep_values else None
        return canon.digest(leaves), kept

    def action(self, leaves: Sequence[np.ndarray]) -> Entry:
        """Digest an action now (before the env can change the array in place)."""
        return self._entry(self.act, leaves)

    def reward(self, reward: Any) -> int:
        return reward_bits(reward, self.reward_dtype)

    def rewards(self, rewards: Any) -> list[int]:
        return reward_bits_batch(rewards, self.reward_dtype)

    def record_reset(self, i: int, obs_leaves: Sequence[np.ndarray]) -> None:
        self.streams[i].add(RESET, self._entry(self.obs, obs_leaves), None, 0, False, False)

    def record_step(
        self,
        i: int,
        action: Entry,
        obs_leaves: Sequence[np.ndarray],
        reward_bits: int,
        terminated: Any,
        truncated: Any,
    ) -> None:
        term = bool(terminated)
        trunc = bool(truncated) and not term
        self.streams[i].add(STEP, self._entry(self.obs, obs_leaves), action, reward_bits, term, trunc)

    def meta(self, **extra: Any) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "stepledger": __version__,
            "reward_dtype": self.reward_dtype.name,
            "observation_space": self.obs.description,
            "action_space": self.act.description,
            "keep_values": self.keep_values,
            "values": {"obs": self.obs.names, "act": self.act.names} if self.keep_values else None,
            **extra,
        }

    def write(self, path: Path, indices: Sequence[int], **extra: Any) -> None:
        """Write every stream recorded so far to ``path`` (atomically replaced)."""
        arrays: dict[str, np.ndarray] = {}
        parts = [stream.columns() for stream in self.streams]
        lengths = [len(stream) for stream in self.streams]
        for name in COLUMNS:
            arrays[name] = np.concatenate([part[name] for part in parts])
        arrays["index"] = np.asarray(indices, dtype=np.int64)
        arrays["offsets"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        arrays["meta"] = np.array(json.dumps(self.meta(**extra)))
        if self.keep_values:
            obs_rows = [row for stream in self.streams for row in stream.obs_values]
            act_rows = [row for stream in self.streams for row in stream.act_values]
            for j, dtype in enumerate(self.obs.dtypes):
                arrays[f"values_obs_{j}"] = _stack([row[j] for row in obs_rows], dtype)
            for j, dtype in enumerate(self.act.dtypes):
                arrays[f"values_act_{j}"] = _stack([None if row is None else row[j] for row in act_rows], dtype)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)


def _stack(values: list[np.ndarray | None], dtype: np.dtype) -> np.ndarray:
    """Stack per-event leaves; missing ones become zeros. Falls back to text if shapes vary."""
    present = [v for v in values if v is not None]
    shapes = {v.shape for v in present}
    if len(shapes) > 1:
        return np.array(["" if v is None else np.array2string(v) for v in values])
    shape = shapes.pop() if shapes else ()
    out = np.zeros((len(values), *shape), dtype=dtype)
    for row, value in enumerate(values):
        if value is not None:
            out[row] = value
    return out


@dataclass
class StreamData:
    """One stream read back from a ledger file."""

    index: int
    columns: dict[str, np.ndarray]
    obs_values: list[np.ndarray] | None  # one array per leaf, rows aligned with columns
    act_values: list[np.ndarray] | None

    def __len__(self) -> int:
        return len(self.columns["kind"])


@dataclass
class LedgerFile:
    path: Path
    meta: dict[str, Any]
    streams: dict[int, StreamData]


def read(path: str | os.PathLike) -> LedgerFile:
    """Read a ledger file written by :class:`Recorder`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: ledger format {meta.get('format')!r}, this stepledger reads format {FORMAT}")
        columns = {name: data[name] for name in COLUMNS}
        indices = data["index"].tolist()
        offsets = data["offsets"].tolist()
        values = meta.get("values") or None
        obs_values = [data[f"values_obs_{j}"] for j in range(len(values["obs"]))] if values else None
        act_values = [data[f"values_act_{j}"] for j in range(len(values["act"]))] if values else None
    streams = {}
    for s, index in enumerate(indices):
        rows = slice(offsets[s], offsets[s + 1])
        streams[index] = StreamData(
            index=index,
            columns={name: col[rows] for name, col in columns.items()},
            obs_values=None if obs_values is None else [v[rows] for v in obs_values],
            act_values=None if act_values is None else [v[rows] for v in act_values],
        )
    return LedgerFile(path=path, meta=meta, streams=streams)


# Ledgers that have not been closed are flushed when the interpreter exits.
_open_ledgers: "weakref.WeakSet[Any]" = weakref.WeakSet()


def track(ledger: Any) -> None:
    _open_ledgers.add(ledger)


def untrack(ledger: Any) -> None:
    _open_ledgers.discard(ledger)


@atexit.register
def _flush_open_ledgers() -> None:
    for ledger in list(_open_ledgers):
        try:
            ledger.flush()
        except Exception as exc:  # never fail interpreter shutdown
            print(f"stepledger: could not flush {getattr(ledger, 'path', ledger)}: {exc}", file=sys.stderr)

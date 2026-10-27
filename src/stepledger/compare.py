"""Compare env-side ledgers with the trainer-side ledger, event by event."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stepledger import _record
from stepledger._canon import reward_from_bits

# (name in reports, column), in the order a step happens: what the trainer sent, then what the env gave.
FIELDS = (
    ("kind", "kind"),
    ("action", "h_act"),
    ("obs", "h_obs"),
    ("reward", "h_rew"),
    ("terminated", "terminated"),
    ("truncated", "truncated"),
)
# Metadata both sides must agree on, or their digests are not comparable.
MUST_AGREE = ("reward_dtype", "observation_space", "action_space")


class MetadataMismatch(ValueError):
    """The two sides recorded with different settings, so their ledgers cannot be compared."""


@dataclass(frozen=True)
class Difference:
    """The first event at which one env's two streams differ.

    ``event`` is the position in both streams; ``episode`` and ``step`` are
    counted on the env side (step 0 is the reset, step k the k-th step).
    """

    index: int
    event: int
    episode: int
    step: int
    fields: tuple[str, ...]
    details: tuple[str, ...] = ()

    def __str__(self) -> str:
        verb = "differs" if len(self.fields) == 1 else "differ"
        text = f"env {self.index}, episode {self.episode}, step {self.step}: {', '.join(self.fields)} {verb}"
        if self.details:
            text += " (" + "; ".join(self.details) + ")"
        return text


@dataclass(frozen=True)
class EnvResult:
    """Outcome for one env index. A count is None when that side has no stream for the env."""

    index: int
    env_events: int | None
    trainer_events: int | None
    difference: Difference | None

    @property
    def clean(self) -> bool:
        return self.difference is None and self.env_events is not None and self.env_events == self.trainer_events

    def lines(self) -> list[str]:
        i, e, t = self.index, self.env_events, self.trainer_events
        if e is None:
            return [f"env {i}: no env ledger (trainer ledger has {_events(t)})"]
        if t is None:
            return [f"env {i}: not in the trainer ledger (env ledger has {_events(e)})"]
        out = [str(self.difference)] if self.difference else []
        if e != t:
            text = f"env {i}: env ledger has {_events(e)}, trainer ledger has {t}"
            if self.difference is None:
                text += f"; the first {min(e, t)} agree"
            out.append(text)
        return out


@dataclass(frozen=True)
class Report:
    """Result of :func:`compare`. ``str(report)`` is a short summary, one line per problem."""

    envs: tuple[EnvResult, ...]
    notes: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return all(env.clean for env in self.envs)

    @property
    def differences(self) -> list[Difference]:
        return [env.difference for env in self.envs if env.difference is not None]

    def __str__(self) -> str:
        lines = list(self.notes)
        for env in self.envs:
            lines += env.lines()
        if self.clean:
            events = sum(env.env_events or 0 for env in self.envs)
            lines.append(f"no differences: {len(self.envs)} envs, {events} events on each side")
        else:
            bad = sum(not env.clean for env in self.envs)
            lines.append(f"{bad} of {len(self.envs)} envs differ")
        return "\n".join(lines)


def compare(env_dir: str | os.PathLike, trainer_path: str | os.PathLike) -> Report:
    """Compare the ``env-*.npz`` ledgers in ``env_dir`` with the trainer ledger at ``trainer_path``.

    Raises :class:`MetadataMismatch` if the two sides recorded with different
    reward dtypes or spaces, and ``FileNotFoundError`` if a side is missing.
    """
    env_dir = Path(env_dir)
    trainer = _record.read(trainer_path)
    _expect_side(trainer, "trainer")
    if not env_dir.is_dir():
        raise FileNotFoundError(f"{env_dir} is not a directory")
    env_files = sorted(env_dir.glob("env-*.npz"))
    if not env_files:
        raise FileNotFoundError(f"no env-*.npz ledgers in {env_dir}")

    env_streams: dict[int, _record.StreamData] = {}
    versions = set()
    for path in env_files:
        ledger = _record.read(path)
        _expect_side(ledger, "env")
        for key in MUST_AGREE:
            if ledger.meta[key] != trainer.meta[key]:
                raise MetadataMismatch(
                    f"{key} differs: {ledger.meta[key]!r} in {path.name}, {trainer.meta[key]!r} in "
                    f"{trainer.path.name}; the two sides must record with the same settings"
                )
        for index, stream in ledger.streams.items():
            if index in env_streams:
                raise ValueError(f"env {index} has more than one env ledger in {env_dir}")
            env_streams[index] = stream
        versions.add(ledger.meta["stepledger"])

    notes = []
    if versions != {trainer.meta["stepledger"]}:
        notes.append(
            f"note: env ledgers written by stepledger {', '.join(sorted(versions))}, "
            f"trainer ledger by {trainer.meta['stepledger']}"
        )
    context = _Context(
        reward_dtype=np.dtype(trainer.meta["reward_dtype"]),
        obs_names=_names(trainer.meta, "obs"),
        act_names=_names(trainer.meta, "act"),
    )
    indices = sorted(set(env_streams) | set(trainer.streams))
    results = tuple(_compare_env(i, env_streams.get(i), trainer.streams.get(i), context) for i in indices)
    return Report(envs=results, notes=tuple(notes))


def _expect_side(ledger: _record.LedgerFile, side: str) -> None:
    if ledger.meta.get("side") != side:
        raise ValueError(f"{ledger.path} is a {ledger.meta.get('side')}-side ledger, expected {side}-side")


def _names(meta: dict, field: str) -> list[str] | None:
    values = meta.get("values")
    return values[field] if values else None


@dataclass(frozen=True)
class _Context:
    reward_dtype: np.dtype
    obs_names: list[str] | None
    act_names: list[str] | None


def _compare_env(
    index: int, env: _record.StreamData | None, trainer: _record.StreamData | None, context: _Context
) -> EnvResult:
    if env is None or trainer is None:
        return EnvResult(index, None if env is None else len(env), None if trainer is None else len(trainer), None)
    n = min(len(env), len(trainer))
    differs = {}
    any_differs = np.zeros(n, dtype=bool)
    for field, column in FIELDS:
        differs[field] = env.columns[column][:n] != trainer.columns[column][:n]
        any_differs |= differs[field]
    hits = np.flatnonzero(any_differs)
    if hits.size == 0:
        return EnvResult(index, len(env), len(trainer), None)
    j = int(hits[0])
    fields = tuple(field for field, _ in FIELDS if differs[field][j])
    if "kind" in fields:  # a reset against a step: the other fields are not comparable
        fields = ("kind",)
    details = tuple(_detail(field, j, env, trainer, context, len(fields) > 1) for field in fields)
    difference = Difference(
        index=index,
        event=j,
        episode=int(env.columns["episode"][j]),
        step=int(env.columns["step"][j]),
        fields=fields,
        details=tuple(d for d in details if d),
    )
    return EnvResult(index, len(env), len(trainer), difference)


def _detail(
    field: str, j: int, env: _record.StreamData, trainer: _record.StreamData, context: _Context, prefix: bool
) -> str:
    """Both sides' values of ``field`` at event ``j``, or "" when they were not kept."""
    if field == "kind":
        name = {_record.RESET: "reset", _record.STEP: "step"}
        return f"env had a {name[int(env.columns['kind'][j])]}, trainer had a {name[int(trainer.columns['kind'][j])]}"
    if field == "reward":
        a = reward_from_bits(int(env.columns["h_rew"][j]), context.reward_dtype)
        b = reward_from_bits(int(trainer.columns["h_rew"][j]), context.reward_dtype)
        text = f"env gave {str(a)}, trainer received {str(b)}"  # str(): f-strings print numpy floats as float64
    elif field in ("terminated", "truncated"):
        text = f"env gave {bool(env.columns[field][j])}, trainer received {bool(trainer.columns[field][j])}"
    else:
        names = context.obs_names if field == "obs" else context.act_names
        env_values = env.obs_values if field == "obs" else env.act_values
        trainer_values = trainer.obs_values if field == "obs" else trainer.act_values
        if names is None or env_values is None or trainer_values is None:
            return ""
        if field == "obs":
            text = _leaves_text(names, [v[j] for v in env_values], [v[j] for v in trainer_values], "env gave", "trainer received")
        else:
            text = _leaves_text(names, [v[j] for v in trainer_values], [v[j] for v in env_values], "trainer sent", "env took")
    return f"{field}: {text}" if prefix else text


def _leaves_text(names: list[str], first: list[np.ndarray], second: list[np.ndarray], said1: str, said2: str) -> str:
    for name, a, b in zip(names, first, second):
        if a.shape != b.shape or a.tobytes() != b.tobytes():
            text = _pair(np.asarray(a), np.asarray(b), said1, said2)
            return f"{name} {text}" if name else text
    return "values equal (digests differ)"


def _pair(a: np.ndarray, b: np.ndarray, said1: str, said2: str) -> str:
    if a.shape != b.shape:
        return f"{said1} shape {a.shape}, {said2} shape {b.shape}"
    if a.size <= 8:
        return f"{said1} {_show(a)}, {said2} {_show(b)}"
    unequal = np.flatnonzero(a.reshape(-1) != b.reshape(-1))
    if unequal.size == 0:
        return "values equal (digests differ)"
    k = int(unequal[0])
    where = tuple(int(w) for w in np.unravel_index(k, a.shape))
    return f"{unequal.size} of {a.size} elements differ, first at {where}: {said1} {_show(a.flat[k])}, {said2} {_show(b.flat[k])}"


def _events(n: int) -> str:
    return "1 event" if n == 1 else f"{n} events"


def _show(a: np.ndarray | np.generic) -> str:
    """Shortest text that tells the value apart in its own dtype (str() of numpy scalars)."""
    if a.ndim == 0:
        return str(a[()])
    if a.ndim == 1:
        return "[" + ", ".join(str(x) for x in a) + "]"
    return np.array2string(a, separator=", ")

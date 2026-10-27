"""Env side: record what one environment took and gave."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

from stepledger import _record


class LedgerEnv(gym.Wrapper):
    """Records every reset and step of ``env`` and writes ``env-{index}.npz`` into ``directory``.

    Put it OUTERMOST in the environment's own process: the last wrapper before
    the vector env, worker or network bridge, so that what it records is
    exactly what crosses the boundary. ``index`` must be the env's position in
    the trainer's vector env.

    The file is written on :meth:`close`, on :meth:`flush`, and at interpreter
    exit for ledgers that were never closed.
    """

    def __init__(
        self,
        env: gym.Env,
        directory: str | os.PathLike,
        index: int,
        reward_dtype: Any = np.float32,
        keep_values: bool = False,
    ) -> None:
        super().__init__(env)
        self.index = int(index)
        self.path = Path(directory) / f"env-{self.index}.npz"
        self._recorder = _record.Recorder(env.observation_space, env.action_space, reward_dtype, keep_values, 1)
        self._closed = False
        _record.track(self)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._recorder.record_reset(0, self._recorder.obs.leaves(obs))
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        rec = self._recorder
        act = rec.action(rec.act.leaves(action))
        obs, reward, terminated, truncated, info = self.env.step(action)
        rec.record_step(0, act, rec.obs.leaves(obs), rec.reward(reward), terminated, truncated)
        return obs, reward, terminated, truncated, info

    def flush(self) -> Path:
        """Write everything recorded so far; returns the file path."""
        self._recorder.write(self.path, [self.index], side="env", index=self.index)
        return self.path

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _record.untrack(self)
            self.flush()
        super().close()

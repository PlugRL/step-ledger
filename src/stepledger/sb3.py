"""Trainer side for Stable-Baselines3 vector environments (needs ``stable-baselines3``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from stepledger import _record


class LedgerVecEnv(VecEnvWrapper):
    """Records what the trainer sent to and received from each sub-environment.

    Put it INNERMOST, directly around ``DummyVecEnv`` / ``SubprocVecEnv`` (or
    the bridge client) and inside ``VecNormalize``, ``VecMonitor``,
    ``VecFrameStack`` and the like, so it sees the values as they arrive.

    SB3 resets an env as soon as its episode ends: on ``done`` the step is
    recorded with ``infos[i]["terminal_observation"]``, then the reset with the
    returned observation. ``truncated`` is ``infos[i]["TimeLimit.truncated"]``.
    """

    def __init__(
        self,
        venv: VecEnv,
        path: str | os.PathLike,
        reward_dtype: Any = np.float32,
        keep_values: bool = False,
    ) -> None:
        super().__init__(venv)
        self.path = Path(path)
        self._recorder = _record.Recorder(
            venv.observation_space, venv.action_space, reward_dtype, keep_values, venv.num_envs
        )
        self._sent: list[_record.Entry] | None = None
        self._closed = False
        _record.track(self)

    def reset(self) -> Any:
        obs = self.venv.reset()
        rec = self._recorder
        leaves = rec.obs.leaves(obs)
        for i in range(self.num_envs):
            rec.record_reset(i, [leaf[i] for leaf in leaves])
        self._sent = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        rec = self._recorder
        leaves = rec.act.leaves(actions)
        self._sent = [rec.action([leaf[i] for leaf in leaves]) for i in range(self.num_envs)]
        self.venv.step_async(actions)

    def step_wait(self) -> Any:
        obs, rewards, dones, infos = self.venv.step_wait()
        if self._sent is None:
            raise RuntimeError("LedgerVecEnv.step_wait() called without step_async()")
        sent, self._sent = self._sent, None
        rec = self._recorder
        obs_leaves = rec.obs.leaves(obs)
        reward_bits = rec.rewards(rewards)
        for i in range(self.num_envs):
            row = [leaf[i] for leaf in obs_leaves]
            if dones[i]:
                truncated = bool(infos[i].get("TimeLimit.truncated", False))
                final_obs = rec.obs.leaves(infos[i]["terminal_observation"])
                rec.record_step(i, sent[i], final_obs, reward_bits[i], not truncated, truncated)
                rec.record_reset(i, row)
            else:
                rec.record_step(i, sent[i], row, reward_bits[i], False, False)
        return obs, rewards, dones, infos

    def flush(self) -> Path:
        """Write everything recorded so far; returns the file path."""
        self._recorder.write(self.path, list(range(self.num_envs)), side="trainer", trainer="sb3")
        return self.path

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _record.untrack(self)
            self.flush()
        self.venv.close()

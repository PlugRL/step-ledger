"""Trainer side for Gymnasium vector environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from stepledger import _record

NEXT_STEP = "NextStep"
SAME_STEP = "SameStep"
DISABLED = "Disabled"


def autoreset_mode(vector_env: gym.vector.VectorEnv) -> str:
    """The vector env's autoreset mode as a string; Gymnasium's default is NextStep."""
    mode = vector_env.metadata.get("autoreset_mode", NEXT_STEP)
    return str(getattr(mode, "value", mode))


class LedgerVectorEnv(gym.vector.VectorWrapper):
    """Records what the trainer sent to and received from each sub-environment.

    Put it INNERMOST on the trainer side, directly around the vector env (or the
    bridge client), so it sees the values as they arrive, before any
    normalization or other wrapper. Handles the NextStep and SameStep autoreset
    modes; Disabled is refused because the wrapper could not tell which resets
    belong to which episode end.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        path: str | os.PathLike,
        reward_dtype: Any = np.float32,
        keep_values: bool = False,
    ) -> None:
        super().__init__(env)
        self.mode = autoreset_mode(env)
        if self.mode == DISABLED:
            raise ValueError(
                "LedgerVectorEnv does not support autoreset_mode=Disabled; "
                "use NextStep or SameStep (gymnasium.vector.AutoresetMode)"
            )
        if self.mode not in (NEXT_STEP, SAME_STEP):
            raise ValueError(f"unknown autoreset mode {self.mode!r}")
        self.path = Path(path)
        self._recorder = _record.Recorder(
            env.single_observation_space, env.single_action_space, reward_dtype, keep_values, env.num_envs
        )
        self._autoreset = np.zeros(env.num_envs, dtype=bool)
        self._closed = False
        _record.track(self)

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        mask = None
        if options is not None and "reset_mask" in options:
            mask = np.array(options["reset_mask"], dtype=bool)  # read before the env pops it
        obs, info = self.env.reset(seed=seed, options=options)
        envs = range(self.num_envs) if mask is None else np.flatnonzero(mask).tolist()
        rec = self._recorder
        leaves = rec.obs.leaves(obs)
        for i in envs:
            rec.record_reset(i, [leaf[i] for leaf in leaves])
            self._autoreset[i] = False
        return obs, info

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        rec = self._recorder
        act_leaves = rec.act.leaves(actions)
        sent = [rec.action([leaf[i] for leaf in act_leaves]) for i in range(self.num_envs)]
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs_leaves = rec.obs.leaves(obs)
        reward_bits = rec.rewards(rewards)
        for i in range(self.num_envs):
            row = [leaf[i] for leaf in obs_leaves]
            if self.mode == NEXT_STEP and self._autoreset[i]:
                # The env was reset; the action sent for it was ignored.
                rec.record_reset(i, row)
            elif self.mode == SAME_STEP and (terminations[i] or truncations[i]):
                final_obs = infos["final_obs"][i]
                rec.record_step(i, sent[i], rec.obs.leaves(final_obs), reward_bits[i], terminations[i], truncations[i])
                rec.record_reset(i, row)
            else:
                rec.record_step(i, sent[i], row, reward_bits[i], terminations[i], truncations[i])
        if self.mode == NEXT_STEP:
            self._autoreset = np.logical_or(terminations, truncations)
        return obs, rewards, terminations, truncations, infos

    def flush(self) -> Path:
        """Write everything recorded so far; returns the file path."""
        self._recorder.write(
            self.path, list(range(self.num_envs)), side="trainer", trainer="gymnasium", autoreset_mode=self.mode
        )
        return self.path

    def close(self, **kwargs: Any) -> Any:
        if not self._closed:
            self._closed = True
            _record.untrack(self)
            self.flush()
        return self.env.close(**kwargs)

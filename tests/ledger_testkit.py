"""Env factories and fault wrappers shared by the tests.

This is a plain module (not a conftest) so that vector-env worker processes can
import the factories by name.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from stepledger import LedgerEnv, LedgerVectorEnv, Report, compare
from stepledger._record import read

# Short time limits so that runs see many episode ends: Pendulum only ever
# truncates; CartPole with random actions mostly terminates, sometimes truncates.
EPISODE_STEPS = {"Pendulum-v1": 25, "CartPole-v1": 30}
FAULTS = ("obs_float16", "stale_reward", "timelimit_as_termination", "action", "skip_step")


class Fault(gym.Wrapper):
    """Changes what crosses the boundary, once.

    Sits between the env-side LedgerEnv and the vector env. It fires at
    ``(episode, step)`` counted the way the trainer sees them (episode 0 starts
    at the first reset, step k is the k-th step of the episode), except
    ``timelimit_as_termination``, which fires at the first truncation.
    """

    def __init__(self, env: gym.Env, kind: str, episode: int, step: int) -> None:
        super().__init__(env)
        if kind not in FAULTS:
            raise ValueError(kind)
        self.kind = kind
        self.at = (episode, step)
        self.episode = -1
        self.t = 0
        self.last: tuple | None = None
        self.fired = False

    def reset(self, **kwargs: Any):
        self.episode += 1
        self.t = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        self.t += 1
        hit = (self.episode, self.t) == self.at
        if hit and self.kind == "skip_step":
            return self.last  # the previous step's result; the env is not stepped
        if hit and self.kind == "action":
            action = other_action(self.action_space, action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if hit and self.kind == "obs_float16":
            obs = round_float16(obs)
        if hit and self.kind == "stale_reward":
            reward = self.last[1]
        if self.kind == "timelimit_as_termination" and truncated and not terminated and not self.fired:
            self.fired = True
            terminated, truncated = True, False
        self.last = (obs, reward, terminated, truncated, info)
        return self.last


def round_float16(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: round_float16(value) for key, value in obs.items()}
    obs = np.asarray(obs)
    return obs.astype(np.float16).astype(obs.dtype)


def other_action(space: spaces.Space, action: Any) -> Any:
    if isinstance(space, spaces.Discrete):
        return np.int64(space.start + (int(action) - space.start + 1) % space.n)
    return (np.asarray(action) + 0.25).astype(space.dtype)


class DictObs(gym.ObservationWrapper):
    """Pendulum with a Dict observation, to exercise nested spaces."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        box = env.observation_space
        self.observation_space = spaces.Dict(
            {
                "velocity": spaces.Box(box.low[2:], box.high[2:], dtype=np.float32),
                "angle": spaces.Box(box.low[:2], box.high[:2], dtype=np.float32),
            }
        )

    def observation(self, obs: np.ndarray) -> dict:
        return {"angle": obs[:2], "velocity": obs[2:]}


def make_env(
    env_id: str,
    directory: str,
    index: int,
    fault: str | None = None,
    at: tuple[int, int] = (1, 7),
    fault_index: int = 1,
    keep_values: bool = False,
    reward_dtype: Any = np.float32,
    dict_obs: bool = False,
) -> gym.Env:
    env = gym.make(env_id, max_episode_steps=EPISODE_STEPS[env_id])
    if dict_obs:
        env = DictObs(env)
    env = LedgerEnv(env, directory, index, reward_dtype=reward_dtype, keep_values=keep_values)  # outermost in-process
    if fault is not None and index == fault_index:
        env = Fault(env, fault, *at)  # between the env-side ledger and the vector env
    return env


def env_fns(env_id: str, directory: str | Path, num_envs: int, **kwargs: Any) -> list:
    return [functools.partial(make_env, env_id, str(directory), i, **kwargs) for i in range(num_envs)]


def run_gymnasium(
    directory: Path,
    env_id: str = "Pendulum-v1",
    vector: str = "sync",
    mode: str = "NextStep",
    num_envs: int = 3,
    steps: int = 100,
    seed: int = 0,
    trainer_kwargs: dict[str, Any] | None = None,
    **env_kwargs: Any,
) -> Report:
    """Run random actions through a ledgered Gymnasium vector env and compare the two sides."""
    fns = env_fns(env_id, directory, num_envs, **env_kwargs)
    cls = SyncVectorEnv if vector == "sync" else AsyncVectorEnv
    vec = LedgerVectorEnv(cls(fns, autoreset_mode=mode), directory / "trainer.npz", **(trainer_kwargs or {}))
    vec.action_space.seed(seed)
    vec.reset(seed=seed)
    for _ in range(steps):
        vec.step(vec.action_space.sample())
    vec.close()
    return compare(directory, directory / "trainer.npz")


def episode_ends(directory: Path) -> tuple[int, int, int]:
    """(resets, terminations, truncations) over all env-side ledgers in ``directory``."""
    resets = terminated = truncated = 0
    for path in directory.glob("env-*.npz"):
        for stream in read(path).streams.values():
            resets += int(np.sum(stream.columns["kind"] == 0))
            terminated += int(np.sum(stream.columns["terminated"]))
            truncated += int(np.sum(stream.columns["truncated"]))
    return resets, terminated, truncated

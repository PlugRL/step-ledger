"""Step ledgers on both sides of a Gymnasium AsyncVectorEnv, then compare.

    python examples/gymnasium_async.py           # clean run: exit code 0
    python examples/gymnasium_async.py --fault   # env 2 hands over one rounded observation: exit code 1
    stepledger compare ledger ledger/trainer.npz
"""

import argparse
import functools

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv

import stepledger


class RoundOnce(gym.Wrapper):
    """Stands in for a boundary that rounds the observation through float16, once, at step 40."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if self.steps == 40:
            obs = obs.astype(np.float16).astype(np.float32)
        return obs, reward, terminated, truncated, info


def make_env(index: int, directory: str, fault: bool) -> gym.Env:
    env = gym.make("Pendulum-v1")
    env = stepledger.LedgerEnv(env, directory, index)  # outermost wrapper in the worker process
    if fault and index == 2:
        env = RoundOnce(env)  # between the env-side ledger and the trainer
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", action="store_true")
    parser.add_argument("--dir", default="ledger")
    args = parser.parse_args()

    envs = AsyncVectorEnv([functools.partial(make_env, i, args.dir, args.fault) for i in range(4)])
    envs = stepledger.LedgerVectorEnv(envs, f"{args.dir}/trainer.npz")  # innermost on the trainer side
    envs.action_space.seed(0)
    envs.reset(seed=0)
    for _ in range(500):
        envs.step(envs.action_space.sample())
    envs.close()  # the workers write env-0.npz ... env-3.npz, the wrapper writes trainer.npz

    report = stepledger.compare(args.dir, f"{args.dir}/trainer.npz")
    print(report)
    raise SystemExit(0 if report.clean else 1)


if __name__ == "__main__":
    main()

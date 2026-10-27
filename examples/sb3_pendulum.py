"""Train PPO on Pendulum with step ledgers on both sides of a SubprocVecEnv, then compare.

    pip install ".[sb3]"   # from a checkout of this repository
    python examples/sb3_pendulum.py
    stepledger compare ledger ledger/trainer.npz
"""

import argparse
import functools

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import stepledger
from stepledger.sb3 import LedgerVecEnv


def make_env(index: int, directory: str) -> gym.Env:
    env = Monitor(gym.make("Pendulum-v1"))
    return stepledger.LedgerEnv(env, directory, index)  # outermost wrapper in the worker process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="ledger")
    parser.add_argument("--timesteps", type=int, default=8192)
    args = parser.parse_args()

    venv = SubprocVecEnv([functools.partial(make_env, i, args.dir) for i in range(4)])
    venv = LedgerVecEnv(venv, f"{args.dir}/trainer.npz")  # innermost on the trainer side
    venv = VecNormalize(venv)  # trainer-side transforms go outside the ledger
    model = PPO("MlpPolicy", venv, n_steps=512, seed=0, verbose=0)
    model.learn(total_timesteps=args.timesteps)
    venv.close()  # the workers write env-0.npz ... env-3.npz, LedgerVecEnv writes trainer.npz

    report = stepledger.compare(args.dir, f"{args.dir}/trainer.npz")
    print(report)
    raise SystemExit(0 if report.clean else 1)


if __name__ == "__main__":
    main()

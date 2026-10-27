"""Time per vector step with and without ledgers, and ledger size per event.

SyncVectorEnv of 16 Pendulum-v1 envs (default 200-step time limit), random
actions sampled up front. Each setting runs ``--repeats`` times, interleaved;
the median is reported.

    python benchmarks/overhead.py
"""

from __future__ import annotations

import argparse
import functools
import platform
import statistics
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv

from stepledger import LedgerEnv, LedgerVectorEnv
from stepledger._record import read

SETTINGS = {
    "no ledgers": (False, False),
    "env side only (LedgerEnv x16)": (True, False),
    "trainer side only (LedgerVectorEnv)": (False, True),
    "both sides": (True, True),
}


def make_env(directory: Path, index: int, env_side: bool, keep_values: bool) -> gym.Env:
    env = gym.make("Pendulum-v1")
    return LedgerEnv(env, directory, index, keep_values=keep_values) if env_side else env


def run(directory: Path, env_side: bool, trainer_side: bool, args: argparse.Namespace, keep_values: bool = False) -> float:
    fns = [functools.partial(make_env, directory, i, env_side, keep_values) for i in range(args.num_envs)]
    vec = SyncVectorEnv(fns)
    if trainer_side:
        vec = LedgerVectorEnv(vec, directory / "trainer.npz", keep_values=keep_values)
    vec.action_space.seed(0)
    actions = [vec.action_space.sample() for _ in range(args.steps)]
    vec.reset(seed=0)
    for action in actions[:200]:  # warm-up
        vec.step(action)
    start = time.perf_counter()
    for action in actions:
        vec.step(action)
    elapsed = time.perf_counter() - start
    vec.close()
    return elapsed / args.steps


def bytes_per_event(directory: Path) -> tuple[float, float]:
    """(env side, trainer side) file bytes per recorded event."""
    env_bytes = env_events = 0
    for path in directory.glob("env-*.npz"):
        env_bytes += path.stat().st_size
        env_events += sum(len(s) for s in read(path).streams.values())
    trainer = directory / "trainer.npz"
    trainer_events = sum(len(s) for s in read(trainer).streams.values())
    return env_bytes / env_events, trainer.stat().st_size / trainer_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print(f"python {platform.python_version()}, numpy {np.__version__}, gymnasium {gym.__version__}, {platform.processor()}")
    times: dict[str, list[float]] = {name: [] for name in SETTINGS}
    with tempfile.TemporaryDirectory() as tmp:
        for r in range(args.repeats):
            for name, (env_side, trainer_side) in SETTINGS.items():
                times[name].append(run(Path(tmp) / f"{r}-{name[:3]}", env_side, trainer_side, args))
        base = statistics.median(times["no ledgers"])
        print(f"\n{args.num_envs} envs, {args.steps} vector steps, median of {args.repeats}")
        for name, values in times.items():
            t = statistics.median(values)
            print(f"  {name:38s} {t * 1e6:8.0f} us/vector step  {t / base:5.2f}x  (+{(t - base) / args.num_envs * 1e6:.1f} us per env step)")

        both = Path(tmp) / "0-bot"
        env_b, trainer_b = bytes_per_event(both)
        print(f"\nfile size per event: env side {env_b:.1f} B, trainer side {trainer_b:.1f} B")
        kept = Path(tmp) / "kept"
        run(kept, True, True, args, keep_values=True)
        env_b, trainer_b = bytes_per_event(kept)
        print(f"with keep_values=True: env side {env_b:.1f} B, trainer side {trainer_b:.1f} B")


if __name__ == "__main__":
    main()

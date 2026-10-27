"""Stable-Baselines3 trainer side. Skipped when stable-baselines3 is not installed."""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from ledger_testkit import env_fns, episode_ends  # noqa: E402
from stepledger import compare  # noqa: E402
from stepledger.sb3 import LedgerVecEnv  # noqa: E402

VEC_ENVS = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}


def run_sb3(directory, env_id, vec, num_envs=3, steps=150, trainer_kwargs=None, **env_kwargs):
    venv = LedgerVecEnv(
        VEC_ENVS[vec](env_fns(env_id, directory, num_envs, **env_kwargs)),
        directory / "trainer.npz",
        **(trainer_kwargs or {}),
    )
    venv.seed(0)
    venv.action_space.seed(0)
    venv.reset()
    for _ in range(steps):
        venv.step(np.stack([venv.action_space.sample() for _ in range(num_envs)]))
    venv.close()
    return compare(directory, directory / "trainer.npz")


@pytest.mark.parametrize("vec", sorted(VEC_ENVS))
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_clean_sb3_run_reports_no_difference(ledger_dir, env_id, vec):
    report = run_sb3(ledger_dir, env_id, vec)
    assert report.clean, str(report)
    resets, terminated, truncated = episode_ends(ledger_dir)
    assert resets > 9 and truncated > 0
    assert (terminated > 0) == (env_id == "CartPole-v1")


@pytest.mark.parametrize("vec", sorted(VEC_ENVS))
def test_sb3_stale_reward_is_found(ledger_dir, vec):
    report = run_sb3(ledger_dir, "Pendulum-v1", vec, steps=100, fault="stale_reward", at=(1, 7))
    assert [d.index for d in report.differences] == [1], str(report)
    diff = report.differences[0]
    assert (diff.episode, diff.step, diff.fields) == (1, 7, ("reward",)), str(report)


@pytest.mark.parametrize("vec", sorted(VEC_ENVS))
def test_sb3_time_limit_sent_as_termination_is_found(ledger_dir, vec):
    report = run_sb3(ledger_dir, "Pendulum-v1", vec, steps=60, fault="timelimit_as_termination")
    diff = report.differences[0]
    assert (diff.index, diff.episode, diff.step) == (1, 0, 25), str(report)
    assert diff.fields == ("terminated", "truncated"), str(report)


def test_sb3_kept_values(ledger_dir):
    report = run_sb3(
        ledger_dir,
        "Pendulum-v1",
        "dummy",
        steps=60,
        fault="obs_float16",
        at=(1, 7),
        keep_values=True,
        trainer_kwargs={"keep_values": True},
    )
    assert str(report.differences[0]).startswith("env 1, episode 1, step 7: obs differs (env gave [")

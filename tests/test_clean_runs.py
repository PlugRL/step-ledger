"""Runs without faults report no difference."""

import numpy as np
import pytest
from gymnasium.vector import SyncVectorEnv

from ledger_testkit import env_fns, episode_ends, run_gymnasium
from stepledger import LedgerVectorEnv, compare


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
@pytest.mark.parametrize("vector", ["sync", "async"])
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_clean_run_reports_no_difference(ledger_dir, env_id, vector, mode):
    report = run_gymnasium(ledger_dir, env_id, vector, mode, num_envs=3, steps=150)
    assert report.clean, str(report)
    assert report.differences == []
    assert [env.index for env in report.envs] == [0, 1, 2]
    assert str(report).startswith("no differences: 3 envs")

    # The run went through several episodes and both kinds of episode end.
    resets, terminated, truncated = episode_ends(ledger_dir)
    assert resets > 3 * 3
    assert truncated > 0
    if env_id == "CartPole-v1":
        assert terminated > 0
    else:
        assert terminated == 0


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
def test_clean_run_with_kept_values_and_dict_observations(ledger_dir, mode):
    report = run_gymnasium(
        ledger_dir, mode=mode, steps=80, keep_values=True, dict_obs=True, trainer_kwargs={"keep_values": True}
    )
    assert report.clean, str(report)


def test_partial_reset_with_reset_mask(ledger_dir):
    vec = LedgerVectorEnv(SyncVectorEnv(env_fns("Pendulum-v1", ledger_dir, 3)), ledger_dir / "trainer.npz")
    vec.reset(seed=0)
    for t in range(60):
        vec.step(vec.action_space.sample())
        if t == 10:
            vec.reset(options={"reset_mask": np.array([True, False, True])})
    vec.close()
    report = compare(ledger_dir, ledger_dir / "trainer.npz")
    assert report.clean, str(report)

"""Faults injected between the env-side ledger and the vector env are found where they happen."""

import re

import pytest

from ledger_testkit import EPISODE_STEPS, run_gymnasium

AT = (1, 7)  # episode 1, step 7: after the first episode end

# fault -> (where the first difference must be, the fields it must name)
EXPECTED = {
    "obs_float16": (AT, ("obs",)),
    "stale_reward": (AT, ("reward",)),
    # Pendulum never terminates, so the first truncation is the end of episode 0.
    "timelimit_as_termination": ((0, EPISODE_STEPS["Pendulum-v1"]), ("terminated", "truncated")),
    # The env took a different action than was sent; what it gave back crossed unchanged.
    "action": (AT, ("action",)),
    "skip_step": (AT, None),  # whatever differs first
}
SETUPS = [("sync", "NextStep"), ("sync", "SameStep"), ("async", "NextStep"), ("async", "SameStep")]


@pytest.mark.parametrize("vector,mode", SETUPS)
@pytest.mark.parametrize("fault", sorted(EXPECTED))
def test_fault_is_found_at_its_env_step_and_field(ledger_dir, fault, vector, mode):
    report = run_gymnasium(ledger_dir, "Pendulum-v1", vector, mode, num_envs=3, steps=100, fault=fault, at=AT)
    assert not report.clean
    assert [d.index for d in report.differences] == [1], str(report)
    assert all(env.clean for env in report.envs if env.index != 1)

    where, fields = EXPECTED[fault]
    diff = report.differences[0]
    assert (diff.episode, diff.step) == where, str(report)
    if fields is not None:
        assert diff.fields == fields, str(report)
    else:  # skip_step: a step whose result was filled in, then one event fewer on the env side
        assert diff.fields and "kind" not in diff.fields, str(report)
        env = report.envs[1]
        assert env.env_events == env.trainer_events - 1, str(report)


def test_report_line_shows_reward_values(ledger_dir):
    report = run_gymnasium(ledger_dir, fault="stale_reward", at=AT)
    line = str(report).splitlines()[0]
    assert re.fullmatch(
        r"env 1, episode 1, step 7: reward differs \(env gave -?[\d.e-]+, trainer received -?[\d.e-]+\)", line
    ), line
    assert str(report).splitlines()[-1] == "1 of 3 envs differ"


def test_kept_values_are_printed_for_observations(ledger_dir):
    report = run_gymnasium(
        ledger_dir, fault="obs_float16", at=AT, keep_values=True, trainer_kwargs={"keep_values": True}
    )
    line = str(report.differences[0])
    assert line.startswith("env 1, episode 1, step 7: obs differs (env gave [")
    assert "trainer received [" in line


def test_kept_values_name_the_leaf_of_a_dict_observation(ledger_dir):
    report = run_gymnasium(
        ledger_dir,
        mode="SameStep",
        fault="obs_float16",
        at=AT,
        keep_values=True,
        dict_obs=True,
        trainer_kwargs={"keep_values": True},
    )
    diff = report.differences[0]
    assert (diff.index, diff.episode, diff.step, diff.fields) == (1, 1, 7, ("obs",))
    assert diff.details[0].startswith("['angle'] env gave [")


def test_changed_discrete_action_is_shown_from_both_sides(ledger_dir):
    report = run_gymnasium(
        ledger_dir,
        "CartPole-v1",
        fault="action",
        at=(0, 3),
        keep_values=True,
        trainer_kwargs={"keep_values": True},
    )
    diff = report.differences[0]
    assert (diff.index, diff.episode, diff.step) == (1, 0, 3)
    assert diff.fields == ("action",)
    assert re.fullmatch(r"trainer sent [01], env took [01]", diff.details[0]), diff.details

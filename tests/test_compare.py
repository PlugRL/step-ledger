"""compare(): metadata checks, envs present on one side only, differing stream lengths, flushing."""

import subprocess
import sys
import textwrap

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from ledger_testkit import env_fns, make_env, run_gymnasium
from stepledger import LedgerEnv, LedgerVectorEnv, MetadataMismatch, compare
from stepledger._record import read


def test_different_reward_dtypes_are_refused(ledger_dir):
    with pytest.raises(MetadataMismatch, match="reward_dtype"):
        run_gymnasium(ledger_dir, reward_dtype=np.float64)


class DeclareFloat16(gym.ObservationWrapper):
    """A boundary that changes the declared observation dtype."""

    def __init__(self, env):
        super().__init__(env)
        box = env.observation_space
        self.observation_space = spaces.Box(box.low.astype(np.float16), box.high.astype(np.float16), dtype=np.float16)

    def observation(self, obs):
        return obs.astype(np.float16)


def test_different_spaces_are_refused(ledger_dir):
    fns = [lambda i=i: DeclareFloat16(make_env("Pendulum-v1", str(ledger_dir), i)) for i in range(2)]
    vec = LedgerVectorEnv(SyncVectorEnv(fns), ledger_dir / "trainer.npz")
    vec.reset(seed=0)
    vec.step(vec.action_space.sample())
    vec.close()
    with pytest.raises(MetadataMismatch, match="observation_space"):
        compare(ledger_dir, ledger_dir / "trainer.npz")


def test_env_on_one_side_only(ledger_dir):
    run_gymnasium(ledger_dir, steps=20)
    (ledger_dir / "env-2.npz").unlink()
    extra = LedgerEnv(gym.make("Pendulum-v1"), ledger_dir, 7)
    extra.reset(seed=0)
    extra.close()

    report = compare(ledger_dir, ledger_dir / "trainer.npz")
    assert not report.clean
    assert report.differences == []
    lines = str(report).splitlines()
    assert lines[0] == "env 2: no env ledger (trainer ledger has 21 events)"
    assert lines[1] == "env 7: not in the trainer ledger (env ledger has 1 event)"
    assert lines[-1] == "2 of 4 envs differ"


def test_extra_event_on_one_side(ledger_dir):
    vec = LedgerVectorEnv(SyncVectorEnv(env_fns("Pendulum-v1", ledger_dir, 2)), ledger_dir / "trainer.npz")
    vec.reset(seed=0)
    for _ in range(5):
        vec.step(vec.action_space.sample())
    vec.env.envs[0].step(np.zeros(1, dtype=np.float32))  # behind the trainer's back
    vec.close()

    report = compare(ledger_dir, ledger_dir / "trainer.npz")
    assert not report.clean
    assert report.differences == []
    assert str(report).splitlines()[0] == "env 0: env ledger has 7 events, trainer ledger has 6; the first 6 agree"


def test_disabled_autoreset_is_refused(ledger_dir):
    vec = SyncVectorEnv(env_fns("Pendulum-v1", ledger_dir, 2), autoreset_mode=AutoresetMode.DISABLED)
    with pytest.raises(ValueError, match="Disabled"):
        LedgerVectorEnv(vec, ledger_dir / "trainer.npz")
    vec.close()


def test_missing_ledgers_and_swapped_arguments(ledger_dir, tmp_path):
    run_gymnasium(ledger_dir, steps=5)
    with pytest.raises(FileNotFoundError):
        compare(tmp_path / "nowhere", ledger_dir / "trainer.npz")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no env-"):
        compare(empty, ledger_dir / "trainer.npz")
    with pytest.raises(ValueError, match="expected trainer-side"):
        compare(ledger_dir, ledger_dir / "env-0.npz")


def test_ledger_file_contents(ledger_dir):
    run_gymnasium(ledger_dir, num_envs=2, steps=30, mode="SameStep")
    env0 = read(ledger_dir / "env-0.npz")
    assert env0.meta["side"] == "env" and env0.meta["index"] == 0
    assert env0.meta["reward_dtype"] == "float32"
    assert env0.meta["observation_space"] == "Box(shape=(3,), dtype=float32)"
    stream = env0.streams[0]
    # reset, 25 steps (truncated at the last), reset, 5 steps
    assert stream.columns["kind"].tolist() == [0] + [1] * 25 + [0] + [1] * 5
    assert stream.columns["episode"].tolist() == [0] * 26 + [1] * 6
    assert stream.columns["step"].tolist() == list(range(26)) + list(range(6))
    assert stream.columns["truncated"].nonzero()[0].tolist() == [25]
    assert not stream.columns["terminated"].any()
    trainer = read(ledger_dir / "trainer.npz")
    assert trainer.meta["side"] == "trainer" and trainer.meta["autoreset_mode"] == "SameStep"
    assert sorted(trainer.streams) == [0, 1]


def test_large_values_are_reported_by_their_first_differing_element():
    from stepledger.compare import _pair

    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = a.copy()
    b[1, 2, 0] = 7
    b[3, 3, 2] = 1
    assert _pair(a, b, "env gave", "trainer received") == (
        "2 of 48 elements differ, first at (1, 2, 0): env gave 0, trainer received 7"
    )
    assert _pair(a, a[:2], "env gave", "trainer received") == "env gave shape (4, 4, 3), trainer received shape (2, 4, 3)"


def test_unclosed_ledger_is_written_at_exit(tmp_path):
    script = textwrap.dedent(
        f"""
        import gymnasium as gym
        from stepledger import LedgerEnv
        env = LedgerEnv(gym.make("Pendulum-v1"), {str(tmp_path)!r}, 4)
        env.reset(seed=0)
        for _ in range(3):
            env.step(env.action_space.sample())
        # no close()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    stream = read(tmp_path / "env-4.npz").streams[4]
    assert stream.columns["kind"].tolist() == [0, 1, 1, 1]

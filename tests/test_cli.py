"""The command line: exit 0 when clean, 1 on any difference, 2 when it cannot compare."""

import subprocess
import sys

from ledger_testkit import run_gymnasium


def cli(*args):
    return subprocess.run([sys.executable, "-m", "stepledger", *map(str, args)], capture_output=True, text=True)


def test_exit_code_0_when_clean(ledger_dir):
    run_gymnasium(ledger_dir, steps=40)
    result = cli("compare", ledger_dir, ledger_dir / "trainer.npz")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("no differences: 3 envs")


def test_exit_code_1_on_a_difference(ledger_dir):
    run_gymnasium(ledger_dir, steps=40, fault="stale_reward", at=(1, 7))
    result = cli("compare", ledger_dir, ledger_dir / "trainer.npz")
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith("env 1, episode 1, step 7: reward differs (env gave ")


def test_exit_code_2_when_it_cannot_compare(tmp_path):
    result = cli("compare", tmp_path / "missing", tmp_path / "trainer.npz")
    assert result.returncode == 2
    assert result.stderr.startswith("stepledger: ")


def test_version():
    result = cli("--version")
    assert result.returncode == 0 and result.stdout.startswith("stepledger ")


def test_base_package_does_not_import_sb3_or_torch():
    code = "import sys, stepledger; assert 'stable_baselines3' not in sys.modules and 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

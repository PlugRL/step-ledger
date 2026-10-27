"""Find silent faults at the environment-trainer boundary by keeping a step ledger on each side.

- :class:`LedgerEnv` (env side, outermost wrapper in the env's process)
- :class:`LedgerVectorEnv` (trainer side, innermost around a Gymnasium vector env)
- ``stepledger.sb3.LedgerVecEnv`` (trainer side, innermost around an SB3 VecEnv)
- :func:`compare` or ``stepledger compare ENV_DIR TRAINER_FILE``
"""

from stepledger._version import __version__
from stepledger._record import LedgerFile, read
from stepledger.compare import Difference, EnvResult, MetadataMismatch, Report, compare
from stepledger.env import LedgerEnv
from stepledger.vector import LedgerVectorEnv

__all__ = [
    "Difference",
    "EnvResult",
    "LedgerEnv",
    "LedgerFile",
    "LedgerVectorEnv",
    "MetadataMismatch",
    "Report",
    "__version__",
    "compare",
    "read",
]


def __getattr__(name: str):
    # ``stepledger.sb3`` imports stable-baselines3 (and torch), so load it only when asked for.
    if name == "sb3":
        import importlib

        return importlib.import_module("stepledger.sb3")
    raise AttributeError(f"module 'stepledger' has no attribute {name!r}")

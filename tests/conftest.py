import sys
from pathlib import Path

import pytest

# Make ledger_testkit importable by the tests and by vector-env worker
# processes, whatever pytest import mode is in use.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def ledger_dir(tmp_path: Path) -> Path:
    return tmp_path / "ledger"

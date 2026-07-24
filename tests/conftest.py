import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).resolve().parent / "data"


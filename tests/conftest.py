import pytest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR

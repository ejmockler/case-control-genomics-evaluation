import pytest
from prefect.testing.utilities import prefect_test_harness
import pandas as pd
import sys

sys.path.append("src")
from example_config import config
from tasks.input import load, processInputFiles


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield

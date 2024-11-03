import pytest

import pathlib


@pytest.fixture
def exp_pardir():
    """Package directory for experiments."""
    return pathlib.Path(__file__).parent / 'experiments'

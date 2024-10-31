import pytest
from hypothesis import settings

import pathlib

settings.register_profile("simplified", max_examples=10)


@pytest.fixture
def exp_pardir():
    """Package directory for experiments."""
    return pathlib.Path(__file__).parent / 'experiments'

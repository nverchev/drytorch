"""Configuration module with shared example settings."""

from typing import Any

import pytest


@pytest.fixture(scope='package')
def example_exp_name() -> str:
    """Example name for the experiment."""
    return 'test_exp'


@pytest.fixture(scope='package',
                params=['test_variation', None],
                ids=['with_variation', 'without_variation'])
def example_variation(request) -> str | None:
    """Yields either a variation name or None."""
    return request.param


@pytest.fixture(scope='package',
                params=[['test_tag'], []],
                ids=['with_tags', 'without_tags'])
def example_tags(request) -> list[str] | None:
    """Yields either a tag or None."""
    return request.param


@pytest.fixture(scope='package')
def example_exp_ts() -> str:
    """Example timestamp for the experiment."""
    return '10_2025-08-01T12-54-21'


@pytest.fixture(scope='package')
def example_config() -> dict[str, int]:
    """Example mapping for metrics."""
    return {
        'batch_size': 32,
        'training_epochs': 100,
    }


@pytest.fixture(scope='package')
def example_source_name() -> str:
    """Example name for the source."""
    return 'test_source'


@pytest.fixture(scope='package')
def example_model_name() -> str:
    """Example name for the model."""
    return 'test_model'


@pytest.fixture(scope='package')
def example_epoch() -> int:
    """Example name for the model."""
    return 5


@pytest.fixture(scope='package')
def example_loss_name() -> str:
    """Example mapping for metrics."""
    return 'Test Loss'


@pytest.fixture(scope='package')
def example_named_metrics(example_loss_name) -> dict[str, float]:
    """Example mapping for metrics."""
    return {
        example_loss_name: 0.456,
        'Accuracy': 0.892,
        'Precision': 0.878,
    }


# tests need primitive types
@pytest.fixture(scope='package')
def example_metadata() -> dict[str, Any]:
    """Example for metadata."""
    return {'architecture': 'ResNet18',
            'long_list': [0] * 20}

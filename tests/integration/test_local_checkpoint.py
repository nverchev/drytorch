"""Tests LocalCheckpoint integration with the Model and ModelOptimizer class."""

from collections.abc import Generator

import pytest

from drytorch.core.experiment import Run
from drytorch.lib.models import ModelOptimizer


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


def test_state_save_and_load(linear_model):
    """Test saving and loading the model's state."""
    param_list = [param.clone() for param in linear_model.module.parameters()]

    linear_model.save_state()

    # change param values
    for param in linear_model.module.parameters():
        param.data.fill_(1)

    # increase epoch
    linear_model.increment_epoch()
    incremented_epoch = linear_model.epoch

    # check params have changed
    for param, old_param in zip(
        linear_model.module.parameters(), param_list, strict=False
    ):
        assert param != old_param

    linear_model.load_state()

    # check original params and epoch
    assert linear_model.epoch < incremented_epoch
    for param, old_param in zip(
        linear_model.module.parameters(), param_list, strict=False
    ):
        assert param == old_param


def test_checkpoint_save_and_load(linear_model, standard_learning_schema):
    """Test saving and loading the model's and the optimizer's states."""
    model_optimizer = ModelOptimizer(linear_model, standard_learning_schema)

    param_list = [param.clone() for param in linear_model.module.parameters()]
    optimizer = model_optimizer._optimizer
    optim_groups = optimizer.param_groups[0].copy()

    model_optimizer.save()

    # change param values
    for param in linear_model.module.parameters():
        param.data.fill_(1)

    # increase epoch
    linear_model.increment_epoch()
    incremented_epoch = linear_model.epoch

    # modify optimizer state
    optimizer.param_groups[0]['lr'] = 1

    # check params have changed
    for param, old_param in zip(
        linear_model.module.parameters(), param_list, strict=False
    ):
        assert param != old_param
    assert optimizer.param_groups[0]['lr'] != optim_groups['lr']

    model_optimizer.load()

    # check original params and epoch
    assert linear_model.epoch < incremented_epoch
    for param, old_param in zip(
        linear_model.module.parameters(), param_list, strict=False
    ):
        assert param == old_param
    assert optimizer.param_groups[0]['lr'] == optim_groups['lr']

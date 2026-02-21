"""Tests LocalCheckpoint integration with the Model subclasses."""

from collections.abc import Generator

import torch

import pytest

from drytorch.core.experimenting import Run
from drytorch.lib.models import SWAModel


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


def test_separate_module_state_save_and_load(run):
    """Test saving and loading SWAModel state."""
    model = SWAModel(torch.nn.Linear(1, 1), start_epoch=0)

    # update average once so it differs
    for p in model.module.parameters():
        p.data.zero_()

    model.post_batch_update()

    base_params = [p.clone() for p in model.module.parameters()]
    avg_params = [p.clone() for p in model.averaged_module.parameters()]

    model.save_state()

    # modify both base and averaged weights
    for p in model.module.parameters():
        p.data.fill_(1)

    for p in model.averaged_module.parameters():
        p.data.fill_(2)

    model.increment_epoch()
    incremented_epoch = model.epoch

    model.load_state()

    # epoch restored
    assert model.epoch < incremented_epoch

    # base restored
    for p, old in zip(model.module.parameters(), base_params, strict=False):
        assert p == old

    # averaged restored
    for p, old in zip(
        model.averaged_module.parameters(), avg_params, strict=False
    ):
        assert p == old


def test_optimizer_save_and_load(linear_model, standard_learning_schema):
    """Test saving and loading the model's and the optimizer's states."""
    optimizer = torch.optim.SGD(linear_model.module.parameters(), lr=0.1)
    linear_model.checkpoint.bind_optimizer(optimizer)
    param_list = [param.clone() for param in linear_model.module.parameters()]
    optim_groups = optimizer.param_groups[0].copy()

    linear_model.checkpoint.save()

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

    linear_model.checkpoint.load()

    # check original params and epoch
    assert linear_model.epoch < incremented_epoch
    for param, old_param in zip(
        linear_model.module.parameters(), param_list, strict=False
    ):
        assert param == old_param

    assert optimizer.param_groups[0]['lr'] == optim_groups['lr']

import numpy as np
import pytest

import torch

from dry_torch import ModelOptimizer
from dry_torch import Experiment
from dry_torch.schedulers import CosineScheduler
from dry_torch.schedulers import ExponentialScheduler
from dry_torch.exceptions import MissingParamError


class ComplexModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear(inputs)))


@pytest.fixture
def complex_model() -> ComplexModel:
    return ComplexModel()


def test_ModelOptimizer_float_lr(complex_model):
    Experiment.new_default_environment().run()
    init_lr = 0.01
    model_optim = ModelOptimizer(complex_model,
                                 lr=init_lr,
                                 other_optimizer_args={'weight_decay': 0.01})
    params = model_optim.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    new_lr = 0.0001
    model_optim.update_learning_rate(new_lr)
    assert params[0]['lr'] == new_lr


def test_ModelOptimizer_dict_lr(complex_model):
    Experiment.new_default_environment().run()
    init_lr = 0.01
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    model_optim = ModelOptimizer(complex_model, lr=lr)
    params = model_optim.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    assert params[1]['lr'] == init_lr / 10
    new_lr = 0.0001
    with pytest.raises(MissingParamError):
        model_optim.update_learning_rate(lr={'linear': new_lr})
    model_optim.epoch = 100
    # constant scheduler should not modify the learning rate
    model_optim.update_learning_rate(lr={'linear': new_lr,
                                         'linear2': new_lr})
    assert params[0]['lr'] == new_lr
    assert params[1]['lr'] == new_lr


def test_ExponentialScheduler(complex_model):
    Experiment.new_default_environment().run()
    init_lr = 0.01
    num_epoch = 100
    decay = 0.99
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    model_optim = ModelOptimizer(
        complex_model,
        lr=lr,
        scheduler=ExponentialScheduler(exp_decay=decay),
    )
    model_optim.update_learning_rate()
    params = model_optim.optimizer.param_groups
    assert np.isclose(params[0]['lr'], init_lr)
    assert np.isclose(params[1]['lr'], init_lr / 10)
    Experiment.get_active_environment().model['model'].epoch = num_epoch
    model_optim.update_learning_rate()
    assert np.isclose(params[0]['lr'], init_lr * decay ** num_epoch)
    assert np.isclose(params[1]['lr'], init_lr / 10 * decay ** num_epoch)


def test_CosineScheduler(complex_model):
    Experiment.new_default_environment().run()
    init_lr = 0.01
    num_epoch = 100
    min_decay = 0.99
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    model_optim = ModelOptimizer(complex_model, lr=lr,
                                 scheduler=CosineScheduler(
                                     decay_steps=num_epoch,
                                     min_decay=min_decay))
    model_optim.update_learning_rate()
    params = model_optim.optimizer.param_groups
    assert np.isclose(params[0]['lr'], init_lr)
    assert np.isclose(params[1]['lr'], init_lr / 10)
    Experiment.get_active_environment().model['model'].epoch = num_epoch
    model_optim.update_learning_rate()
    assert np.isclose(params[0]['lr'], init_lr * min_decay)
    assert np.isclose(params[1]['lr'], init_lr / 10 * min_decay)

    Experiment.get_active_environment().model['model'].epoch = 2 * num_epoch
    model_optim.update_learning_rate()
    assert np.isclose(params[0]['lr'], init_lr * min_decay)
    assert np.isclose(params[1]['lr'], init_lr / 10 * min_decay)


@pytest.mark.parametrize("clone", [False, True])
def test_model_optimizer(complex_model, clone) -> None:
    Experiment.new_default_environment().run()
    init_lr = 0.01
    model_optim = ModelOptimizer(complex_model, lr=0.01)

    if clone:
        model_optim = model_optim.clone('clone')

    model_params = {id(param) for param in model_optim.model.parameters()}
    optimizer_params = {id(param) for group in
                        model_optim.optimizer.param_groups for param in
                        group['params']}
    assert model_params == optimizer_params
    assert model_optim.optimizer.param_groups[0]['lr'] == init_lr

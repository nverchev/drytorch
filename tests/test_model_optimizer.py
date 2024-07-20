import numpy as np
import pytest

import torch

from dry_torch import Model
from dry_torch import LearningScheme
from dry_torch import Experiment
from dry_torch import register_model
from dry_torch.scheduling import CosineScheduler
from dry_torch.scheduling import ExponentialScheduler
from dry_torch.exceptions import MissingParamError
from dry_torch.learning import ModelOptimizer


class ComplexModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear(inputs)))


@pytest.fixture(scope='module')
def _complex_model() -> Model[torch.Tensor, torch.Tensor]:
    complex_model = Model(ComplexModel(), name='complex_model')
    register_model(complex_model)
    return complex_model


@pytest.fixture()
def complex_model(_complex_model) -> Model[torch.Tensor, torch.Tensor]:
    Experiment.current().tracker['complex_model'].epoch = 0
    return _complex_model


def test_Model_float_lr(complex_model) -> None:
    init_lr = 0.01
    model_optimizer = ModelOptimizer(complex_model,
                                     LearningScheme(lr=init_lr))
    params = model_optimizer.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    new_lr = 0.0001
    model_optimizer.update_learning_rate(new_lr)
    assert params[0]['lr'] == new_lr


def test_Model_dict_lr(complex_model) -> None:
    init_lr = 0.01
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    model_optimizer = ModelOptimizer(complex_model,
                                     LearningScheme(lr=lr))
    params = model_optimizer.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    assert params[1]['lr'] == init_lr / 10
    new_lr = 0.0001
    with pytest.raises(MissingParamError):
        model_optimizer.update_learning_rate(lr={'linear': new_lr})
    Experiment.current().tracker['complex_model'].epoch = 100
    # constant scheduler should not modify the learning rate
    model_optimizer.update_learning_rate(lr={'linear': new_lr,
                                             'linear2': new_lr})
    assert params[0]['lr'] == new_lr
    assert params[1]['lr'] == new_lr


def test_ExponentialScheduler(complex_model) -> None:
    init_lr = 0.01
    num_epoch = 100
    decay = 0.99
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    learning_scheme = LearningScheme(
        lr=lr,
        scheduler=ExponentialScheduler(exp_decay=decay),
    )
    model_optimizer = ModelOptimizer(complex_model, learning_scheme)

    model_optimizer.update_learning_rate()
    params = model_optimizer.optimizer.param_groups
    assert np.isclose(params[0]['lr'], init_lr)
    assert np.isclose(params[1]['lr'], init_lr / 10)
    Experiment.current().tracker['complex_model'].epoch = num_epoch
    model_optimizer.update_learning_rate()
    assert np.isclose(params[0]['lr'], init_lr * decay ** num_epoch)
    assert np.isclose(params[1]['lr'], init_lr / 10 * decay ** num_epoch)


def test_CosineScheduler(complex_model) -> None:
    init_lr = 0.01
    num_epoch = 100
    min_decay = 0.99
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    learning_scheme = LearningScheme(
        lr=lr,
        scheduler=CosineScheduler(decay_steps=num_epoch, min_decay=min_decay),
    )
    model_optimizer = ModelOptimizer(complex_model, learning_scheme)
    model_optimizer.update_learning_rate()
    params = model_optimizer.optimizer.param_groups
    assert np.isclose(params[0]['lr'], init_lr)
    assert np.isclose(params[1]['lr'], init_lr / 10)
    Experiment.current().tracker['complex_model'].epoch = num_epoch
    model_optimizer.update_learning_rate()
    assert np.isclose(params[0]['lr'], init_lr * min_decay)
    assert np.isclose(params[1]['lr'], init_lr / 10 * min_decay)

    Experiment.current().tracker['complex_model'].epoch = 2 * num_epoch
    model_optimizer.update_learning_rate()
    assert np.isclose(params[0]['lr'], init_lr * min_decay)
    assert np.isclose(params[1]['lr'], init_lr / 10 * min_decay)


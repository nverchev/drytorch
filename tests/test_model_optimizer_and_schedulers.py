import pytest

import torch

from dry_torch.model_optimizer import ModelOptimizer
from dry_torch.protocols import TypedModule
from dry_torch.schedulers import CosineScheduler, ExponentialScheduler


class ComplexModel(TypedModule[torch.Tensor, torch.Tensor]):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear(inputs)))


@pytest.fixture
def complex_model() -> TypedModule[torch.Tensor, torch.Tensor]:
    return ComplexModel()


def test_ModelOptimizer_float_lr():
    model = torch.nn.Linear(1, 1)
    init_lr = 0.01
    model_optim: ModelOptimizer = ModelOptimizer(model, lr=init_lr, other_optimizer_args={'weight_decay': 0.000001})
    params = model_optim.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    new_lr = 0.0001
    model_optim.update_learning_rate(new_lr)
    assert params[0]['lr'] == new_lr


def test_ModelOptimizer_dict_lr(complex_model):
    init_lr = 0.01
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    model_optim: ModelOptimizer[torch.Tensor, torch.Tensor] = ModelOptimizer(complex_model, lr=lr)
    params = model_optim.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    assert params[1]['lr'] == init_lr / 10
    new_lr = 0.0001
    with pytest.raises(ValueError):
        model_optim.update_learning_rate(lr={'linear': new_lr})
    model_optim.epoch = 100  # constant scheduler should not modify the learning rate
    model_optim.update_learning_rate(lr={'linear': new_lr, 'linear2': new_lr})
    assert params[0]['lr'] == new_lr
    assert params[1]['lr'] == new_lr


def test_ExponentialScheduler(complex_model):
    init_lr = 0.01
    num_epoch = 100
    decay = 0.99
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    model_optim = ModelOptimizer(complex_model, lr=lr, scheduler=ExponentialScheduler(exp_decay=decay))
    model_optim.update_learning_rate()
    params = model_optim.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    assert params[1]['lr'] == init_lr / 10
    model_optim.epoch = num_epoch
    model_optim.update_learning_rate()
    assert params[0]['lr'] == init_lr * decay ** num_epoch
    assert params[1]['lr'] == init_lr / 10 * decay ** num_epoch


def test_CosineScheduler(complex_model):
    init_lr = 0.01
    num_epoch = 100
    min_decay = 0.99
    lr = {'linear': init_lr, 'linear2': init_lr / 10}
    model_optim = ModelOptimizer(complex_model, lr=lr, scheduler=CosineScheduler(decay_steps=num_epoch,
                                                                                 min_decay=min_decay))
    model_optim.update_learning_rate()
    params = model_optim.optimizer.param_groups
    assert params[0]['lr'] == init_lr
    assert params[1]['lr'] == init_lr / 10
    model_optim.epoch = num_epoch
    model_optim.update_learning_rate()
    assert params[0]['lr'] == init_lr * min_decay
    assert params[1]['lr'] == init_lr / 10 * min_decay
    model_optim.epoch = 2 * num_epoch
    model_optim.update_learning_rate()
    assert params[0]['lr'] == init_lr * min_decay
    assert params[1]['lr'] == init_lr / 10 * min_decay

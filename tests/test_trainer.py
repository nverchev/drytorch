import logging

import pytest
import torch
from torch.utils import data
from dry_torch import Trainer
from dry_torch import StandardLoader
from dry_torch import Experiment
from dry_torch import ModelOptimizer
from dry_torch import LossAndMetricsCalculator
from dry_torch import exceptions
from dry_torch import default_logging


class IdentityDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor([index]) / len(self)
        return x, x

    def __len__(self) -> int:
        return 1600


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def square_error(tensor: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (tensor - second) ** 2


logger = logging.getLogger('dry_torch')
logger.setLevel(default_logging.INFO_LEVELS.progress_bar)


def test_all() -> None:
    exp_pardir = 'test_experiments'
    Experiment('test_simple_training',
               exp_pardir=exp_pardir,
               config={'answer': 42}).run()
    model = Linear(1, 1)
    model_opt = ModelOptimizer(model, lr=0.1)
    cloned_model_opt = model_opt.clone('cloned_model')
    loss_calc = LossAndMetricsCalculator(square_error)
    dataset = IdentityDataset()
    loader = StandardLoader(dataset=dataset, batch_size=4)
    trainer = Trainer(cloned_model_opt,
                      loss_calc=loss_calc,
                      train_loader=loader,
                      val_loader=loader)
    trainer.train(10)
    cloned_model_opt.save()
    Trainer(model_opt, loss_calc=loss_calc, train_loader=loader)
    with pytest.raises(exceptions.AlreadyBoundedError):
        Trainer(model_opt, loss_calc=loss_calc, train_loader=loader)
    out = cloned_model_opt.model(torch.FloatTensor([.2]).to(model_opt.device))
    print(out)
    assert torch.isclose(out, torch.tensor(.2), atol=0.001)


if __name__ == '__main__':
    test_all()

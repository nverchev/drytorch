import logging
import pathlib

import torch
from torch.utils import data
from dry_torch import Trainer
from dry_torch import DataLoader
from dry_torch import Experiment
from dry_torch import Model
from dry_torch import SimpleLossCalculator
from dry_torch import LearningScheme
from dry_torch import register_model
from dry_torch import hooks
from typing import NamedTuple
import dataclasses


class TorchTuple(NamedTuple):
    input: torch.Tensor


@dataclasses.dataclass()
class TorchData:
    output: torch.Tensor
    output2: tuple[torch.Tensor, ...] = (torch.empty(0),)


class IdentityDataset(data.Dataset[tuple[TorchTuple, torch.Tensor]]):

    def __init__(self):
        self.tensors = torch.rand(7, 7, 7)
        super().__init__()

    def __getitem__(self, index: int) -> tuple[
        TorchTuple, torch.Tensor
    ]:
        x = torch.FloatTensor([index]) / len(self)
        return TorchTuple(x), x

    def __len__(self) -> int:
        return 1600


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs: TorchTuple) -> TorchData:
        return TorchData(self.linear(inputs.input))


def square_error(outputs: TorchData,
                 targets: torch.Tensor) -> torch.Tensor:
    return torch.stack(2 * [(outputs.output - targets) ** 2]).mean()


def test_all() -> None:
    exp_pardir = pathlib.Path(__file__).parent / 'experiments'

    Experiment('test_sweep',
               pardir=exp_pardir)

    loss_calc = SimpleLossCalculator(loss_fun=square_error)
    dataset = IdentityDataset()
    loader = DataLoader(dataset=dataset, batch_size=4)
    callback = hooks.early_stopping_callback(patience=0, start_from_epoch=5)

    for i in range(5):
        module = Linear(1, 1)
        model = Model(module)
        trainer = Trainer(model,
                          learning_scheme=LearningScheme(lr=i * 0.01),
                          loss_calc=loss_calc,
                          loader=loader)
        trainer.add_validation(val_loader=loader)
        trainer.post_epoch_hooks.register(callback)
        trainer.train(10)

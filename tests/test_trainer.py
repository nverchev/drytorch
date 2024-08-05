import logging
import pathlib

import pytest
import torch
from torch.utils import data
from dry_torch import Trainer
from dry_torch import Test as _Test  # otherwise pytest interprets it as a test
from dry_torch import DataLoader
from dry_torch import Experiment
from dry_torch import Model
from dry_torch import SimpleLossCalculator
from dry_torch import exceptions
from dry_torch import LearningScheme
from dry_torch import protocols as p
from dry_torch import register_model
from dry_torch import log_settings
from dry_torch import hooks
from typing import NamedTuple, Iterable
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

    Experiment('test_simple_training',
               pardir=exp_pardir)

    module = Linear(1, 1)
    loss_calc = SimpleLossCalculator(loss_fun=square_error)
    model = Model(module, name='original_model')
    register_model(model)
    dataset = IdentityDataset()
    loader = DataLoader(dataset=dataset, batch_size=4)
    trainer = Trainer(model,
                      name='MyTrainer',
                      learning_scheme=LearningScheme(lr=0.01),
                      loss_calc=loss_calc,
                      loader=loader)
    trainer.add_validation(val_loader=loader)
    trainer.post_epoch_hooks.register(
        hooks.call_every(5, hooks.saving_hook())
    )
    trainer.post_epoch_hooks.register(hooks.early_stopping_callback())
    trainer.train(5)
    trainer.save_checkpoint()
    trainer.load_checkpoint()
    trainer.train(5)
    cloned_model = model.clone('cloned_model')
    register_model(cloned_model)
    Trainer(cloned_model,
            learning_scheme=LearningScheme(lr=0.01),
            loss_calc=loss_calc,
            loader=loader)
    tuple_in = TorchTuple(input=torch.FloatTensor([.2]).to(cloned_model.device))
    out = cloned_model(tuple_in)
    assert torch.isclose(out.output, torch.tensor(.2), atol=0.01)
    trainer.terminate_training()
    test = _Test(model,
                 metrics_calc=loss_calc,
                 loader=loader,
                 )
    test(store_outputs=True)


if __name__ == "__main__":
    test_all()

import pathlib
import time

import torch

from torch.utils import data

from src import dry_torch
from src.dry_torch import Trainer
from src.dry_torch import Test as _Test  # pytest interprets Test as a test
from src.dry_torch import DataLoader
from src.dry_torch import Experiment
from src.dry_torch import Model
from src.dry_torch import SimpleLossCalculator, MetricsCalculator
from src.dry_torch import LearningScheme
from src.dry_torch import hooks
from src.dry_torch import set_compact_mode
from typing import NamedTuple
import dataclasses


class TorchTuple(NamedTuple):
    input: torch.Tensor


set_compact_mode()


@dataclasses.dataclass()
class TorchData:
    output: torch.Tensor
    output2: tuple[torch.Tensor, ...] = (torch.empty(0),)


class IdentityDataset(data.Dataset[tuple[TorchTuple, torch.Tensor]]):

    def __init__(self):
        self.ones = [1, 1, 1, 1, 1] * 5
        self.torch_ones = torch.ones(3, 3, 3)
        super().__init__()

    def __getitem__(self, index: int) -> tuple[
        TorchTuple, torch.Tensor
    ]:
        time.sleep(0.001)
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
    return ((outputs.output - targets) ** 2).mean() + torch.rand([1]).to(
        'cuda')


def zero(outputs: TorchData,
         targets: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0)


def test_all() -> None:
    exp_pardir = pathlib.Path(__file__).parent / 'experiments'

    Experiment('test_simple_training', pardir=exp_pardir).start()
    module = Linear(1, 1)
    loss_calc = SimpleLossCalculator(loss_fun=square_error)
    metrics_calc = MetricsCalculator(my_metric=square_error, zero=zero)

    model = Model(module, name='original_model')
    dataset = IdentityDataset()
    loader = DataLoader(dataset=dataset, batch_size=4)
    trainer = Trainer(model,
                      name='MyTrainer',
                      learning_scheme=LearningScheme(torch.optim.Adam, lr=0.01),
                      loss_calc=loss_calc,
                      loader=loader)

    trainer.add_validation(val_loader=loader)
    trainer.post_epoch_hooks.register(
        hooks.early_stopping_callback(
            monitor_validation=False,
            monitor_external=trainer.validation,
            patience=1))
    trainer.post_epoch_hooks.register(
        hooks.call_every(5, hooks.saving_hook())
    )

    trainer.train(3)
    trainer.train(3)

    cloned_model = model.clone('cloned_model')
    Trainer(cloned_model,
            learning_scheme=LearningScheme(torch.optim.Adam, lr=0.01),
            loss_calc=loss_calc,
            loader=loader)
    tuple_in = TorchTuple(input=torch.FloatTensor([.2]).to(cloned_model.device))
    out = cloned_model(tuple_in)
    assert torch.isclose(out.output, torch.tensor(.2), atol=0.01)
    test = _Test(model,
                 metrics_calc=metrics_calc,
                 loader=loader,
                 )
    test(store_outputs=True)


if __name__ == "__main__":
    test_all()

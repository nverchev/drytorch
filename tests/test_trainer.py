import logging

import pytest
import torch
from torch.utils import data
from dry_torch import Trainer
from dry_torch import Test as _Test  # other pytest interprets it as a test
from dry_torch import DataLoader
from dry_torch import Experiment
from dry_torch import Model
from dry_torch import LossCalculator
from dry_torch import exceptions
from dry_torch import default_logging
from dry_torch import LearningScheme

from typing import NamedTuple


class TorchTuple(NamedTuple):
    input: torch.Tensor


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

    def forward(self, inputs: TorchTuple) -> torch.Tensor:
        return self.linear(inputs.input)


def square_error(outputs: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
    return (outputs - targets) ** 2


logger = logging.getLogger('dry_torch')
logger.setLevel(default_logging.INFO_LEVELS.tqdm_bar)


def test_all() -> None:
    exp_pardir = 'test_experiments'
    Experiment('test_simple_training',
               exp_pardir=exp_pardir,
               config={'answer': 42})
    module = Linear(1, 1)
    loss_calc = LossCalculator(loss_fun=square_error)
    model = Model(module, name='original_model')
    dataset = IdentityDataset()
    loader = DataLoader(dataset=dataset, batch_size=4)
    trainer = Trainer(model,
                      learning_scheme=LearningScheme(lr=0.01),
                      calculator=loss_calc,
                      loader=loader,
                      val_loader=loader)
    trainer.train(10)
    trainer.save_checkpoint()
    cloned_model = model.clone('cloned_model')
    Trainer(cloned_model,
            learning_scheme=LearningScheme(lr=0.01),
            calculator=loss_calc,
            loader=loader,
            val_loader=loader)
    with pytest.raises(exceptions.AlreadyBoundedError):
        Trainer(cloned_model,
                learning_scheme=LearningScheme(lr=0.01),
                calculator=loss_calc,
                loader=loader,
                val_loader=loader)
    sample_in = TorchTuple(
        input=torch.FloatTensor([.2]).to(cloned_model.device)
    )
    out = cloned_model(sample_in)
    assert torch.isclose(out, torch.tensor(.2), atol=0.01)

    test = _Test(model,
                 calculator=loss_calc,
                 loader=loader,
                 store_outputs=True)
    test()
    with pytest.warns(exceptions.AlreadyTestedWarning):
        test()


if __name__ == '__main__':
    test_all()

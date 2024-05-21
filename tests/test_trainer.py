import pytest
import torch
from torch.utils.data import Dataset
from dry_torch import Trainer
from dry_torch import Split
from dry_torch import Loaders
from dry_torch import Experiment
from dry_torch import ModelOptimizer
from dry_torch import CheckpointIO
from dry_torch import LossAndMetricsCalculator
from dry_torch import exceptions


class IndexDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.Tensor([index]), torch.Tensor([index])

    def __len__(self) -> int:
        return 1600


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def simple_fun(tensor: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return tensor + second


def test_all() -> None:
    Experiment('test_simple_training', config={'answer': 42}).run()
    exp_pardir = 'test_experiments'
    model = Linear(1, 1)
    model_opt = ModelOptimizer(model)
    cloned_model_opt = model_opt.clone('cloned_model')
    checkpoint = CheckpointIO(model_opt, exp_pardir=exp_pardir)
    loss_calc = LossAndMetricsCalculator(simple_fun)
    dataset = IndexDataset()
    loaders = Loaders(train_dataset=dataset,
                      test_dataset=dataset,
                      batch_size=4)

    trainer = Trainer(cloned_model_opt, loss_calc=loss_calc, loaders=loaders)
    trainer.train(2)
    trainer.test(partition=Split.TEST, save_outputs=True)
    checkpoint.save()
    Trainer(model_opt, loss_calc=loss_calc, loaders=loaders)
    with pytest.raises(exceptions.AlreadyBoundedError):
        Trainer(model_opt, loss_calc=loss_calc, loaders=loaders)

import pytest
from hypothesis import settings

import pathlib

import torch

from src.dry_torch import DataLoader, MetricsCalculator, Model
from src.dry_torch import SimpleLossCalculator
from src.dry_torch import protocols as p
from tests.example_classes import TorchTuple, TorchData, IdentityDataset, Linear

settings.register_profile("simplified", max_examples=10)


def mse(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
    """Mean square error calculation."""
    return ((outputs.output - targets) ** 2).mean()


@pytest.fixture
def exp_pardir():
    """Package directory for experiments."""
    return pathlib.Path(__file__).parent / 'experiments'


@pytest.fixture
def linear_model() -> p.ModelProtocol[TorchTuple, TorchData]:
    """Provides a simple linear model with 1 input and 1 output feature."""
    return Model(Linear(1, 1), name='linear')


@pytest.fixture
def identity_dataset() -> IdentityDataset:
    """Provides the IdentityDataset instance."""
    return IdentityDataset()


@pytest.fixture
def identity_loader(
        identity_dataset
) -> p.LoaderProtocol[tuple[TorchTuple, torch.Tensor]]:
    """Provides the IdentityDataset instance."""
    return DataLoader(dataset=identity_dataset, batch_size=4)


@pytest.fixture
def square_calc() -> p.LossCalculatorProtocol[TorchData, torch.Tensor]:
    """Provides the IdentityDataset instance."""
    return SimpleLossCalculator(loss_fun=mse)


@pytest.fixture
def metrics_calc() -> p.MetricsCalculatorProtocol[TorchData, torch.Tensor]:
    """Provides the IdentityDataset instance."""

    def zero(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
        """Dummy metric calculation."""
        _not_used = outputs, targets
        return torch.tensor(0)

    return MetricsCalculator(Criterion=mse, Zero=zero)

import pytest

import torch

from src.dry_torch import DataLoader, LearningScheme, MetricsCalculator, Model
from src.dry_torch import SimpleLossCalculator, Trainer
from src.dry_torch import protocols as p

from tests.integration.example_classes import IdentityDataset, Linear
from tests.integration.example_classes import TorchTuple, TorchData


@pytest.fixture
def linear_model() -> p.ModelProtocol[TorchTuple, TorchData]:
    return Model(Linear(1, 1), name='linear')


@pytest.fixture
def identity_dataset() -> IdentityDataset:
    return IdentityDataset()


@pytest.fixture
def identity_loader(
        identity_dataset
) -> p.LoaderProtocol[tuple[TorchTuple, torch.Tensor]]:
    return DataLoader(dataset=identity_dataset, batch_size=4)


@pytest.fixture
def square_loss_calc() -> p.LossCalculatorProtocol[TorchData, torch.Tensor]:
    def mse(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
        """Mean square error calculation."""
        return ((outputs.output - targets) ** 2).mean()

    return SimpleLossCalculator(loss_fun=mse)


@pytest.fixture
def zero_metrics_calc() -> p.MetricsCalculatorProtocol[TorchData, torch.Tensor]:
    def zero(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
        """Dummy metric calculation."""
        _not_used = outputs, targets
        return torch.tensor(0)

    return MetricsCalculator(Zero=zero)


@pytest.fixture
def example_learning_scheme() -> p.LearningProtocol:
    return LearningScheme(torch.optim.Adam, lr=0.01)


@pytest.fixture
def identity_trainer(linear_model,
                     example_learning_scheme,
                     square_loss_calc,
                     identity_loader) -> p.TrainerProtocol:
    return Trainer(linear_model,
                   name='MyTrainer',
                   loader=identity_loader,
                   learning_scheme=example_learning_scheme,
                   loss_calc=square_loss_calc)

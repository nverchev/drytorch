"""Configuration module with objects from the package."""

from collections.abc import Generator

import torch

import pytest

import drytorch

from drytorch import (
    DataLoader,
    Experiment,
    LearningSchema,
    Loss,
    Metric,
    Model,
    Trainer,
)
from drytorch.core.experiment import Run
from tests.simple_classes import IdentityDataset, Linear, TorchData, TorchTuple


@pytest.fixture
def linear_model() -> Model[TorchTuple, TorchData]:
    """Instantiate a simple model."""
    return Model(Linear(1, 1), name='linear')


@pytest.fixture
def identity_dataset() -> IdentityDataset:
    """Instantiate a simple dataset."""
    return IdentityDataset()


@pytest.fixture
def identity_loader(
    identity_dataset,
) -> DataLoader[tuple[TorchTuple, torch.Tensor]]:
    """Instantiate a loader for the identity dataset."""
    return DataLoader(dataset=identity_dataset, batch_size=4)


@pytest.fixture
def zero_metrics_calc() -> Metric[TorchData, torch.Tensor]:
    """Instantiate a null metric for the identity dataset."""

    def zero(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
        """Fake metric calculation from structured outputs.

        Args:
            outputs: structured model outputs.
            targets: tensor for the ground truth.

        Returns:
            zero tensor.
        """
        _not_used = outputs, targets
        return torch.tensor(0)

    return Metric(zero, name='Zero', higher_is_better=True)


@pytest.fixture
def square_loss_calc() -> Loss[TorchData, torch.Tensor]:
    """Instantiate a loss for the identity dataset."""

    def mse(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
        """Mean square error calculation from structured outputs.

        Args:
            outputs: structured model outputs.
            targets: tensor for the ground truth.

        Returns:
            mean square error.
        """
        return ((outputs.output - targets) ** 2).mean()

    return Loss(mse, 'MSE')


@pytest.fixture
def standard_learning_schema() -> LearningSchema:
    """Instantiate a standard learning scheme."""
    return LearningSchema.adam(base_lr=0.1)


@pytest.fixture
def identity_trainer(
    linear_model,
    standard_learning_schema: LearningSchema,
    square_loss_calc: Loss[TorchData, torch.Tensor],
    identity_loader: DataLoader[tuple[TorchTuple, torch.Tensor]],
) -> Trainer[TorchTuple, torch.Tensor, TorchData]:
    """Instantiate a trainer for the linear model using the identity dataset."""
    trainer = Trainer(
        linear_model,
        name='MyTrainer',
        loader=identity_loader,
        learning_schema=standard_learning_schema,
        loss=square_loss_calc,
    )
    return trainer


@pytest.fixture(scope='module')
def run(tmpdir_factory, example_run_id) -> Generator[Run, None, None]:
    """Fixture of an experiment."""
    drytorch.remove_all_default_trackers()
    par_dir = tmpdir_factory.mktemp('experiments')
    exp = Experiment(name='TestExperiment', par_dir=par_dir, config=None)
    with exp.create_run(run_id=example_run_id) as run:
        yield run
    return

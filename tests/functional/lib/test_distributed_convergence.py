"""DDP functional tests for Trainer on CPU."""

import torch

import pytest

from drytorch import (
    DataLoader,
    LearningSchema,
    Loss,
    Model,
    Trainer,
)
from tests.functional.conftest import DistributedWorker, RunningWorker
from tests.simple_classes import IdentityDataset, Linear, TorchData, TorchTuple


WORLD_SIZE = 2


def mse(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
    """Mean square error calculation from structured outputs."""
    return ((outputs.output - targets) ** 2).mean()


def setup_training() -> Trainer[TorchTuple, torch.Tensor, TorchData]:
    """Setup DDP training."""
    network = Linear(1, 1)
    model = Model(network, name='linear', device=torch.device('cpu'))
    dataset = IdentityDataset(80)
    loader = DataLoader(dataset, batch_size=4)
    loss = Loss(mse, name='MSE')
    learning_schema = LearningSchema.sgd(momentum=0)
    return Trainer(
        model,
        learning_schema=learning_schema,
        loader=loader,
        loss=loss,
        name='MyDDPTrainer',
    )


def assert_convergence() -> None:
    """Assert convergence while using DDP."""
    trainer = setup_training()
    trainer.train(1)
    initial_loss = trainer.computed_metrics
    trainer.train(2)
    final_loss = trainer.computed_metrics
    metric_name = trainer.objective.name

    assert final_loss[metric_name] < initial_loss[metric_name]


@pytest.mark.parametrize('world_size', [WORLD_SIZE])
def test_ddp_convergence(example_run_id, tmp_path, world_size) -> None:
    """Test convergence of DDP training."""
    running_worker = RunningWorker(
        assert_convergence, par_dir=tmp_path, run_id=example_run_id
    )
    worker = DistributedWorker(running_worker, world_size=world_size)
    exit_codes, _ = worker.process()

    assert all(exit_code == 0 for exit_code in exit_codes)

"""DDP functional tests for Trainer on CPU."""

import sys

import torch

from ...simple_classes import IdentityDataset, Linear, TorchData, TorchTuple
from ..conftest import DistributedWorker, RunningWorker
from torch import nn

import pytest

from drytorch import (
    DataLoader,
    LearningSchema,
    Loss,
    Model,
    Trainer,
)


WORLD_SIZE = 2


def mse(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
    """Mean square error calculation from structured outputs."""
    return ((outputs.output - targets) ** 2).mean()


def setup_training() -> Trainer[TorchTuple, torch.Tensor, TorchData]:
    """Setup DDP training."""
    network = Linear(1, 1)
    ddp_network = nn.parallel.DistributedDataParallel(network)
    model = Model(ddp_network, name='linear')
    dataset = IdentityDataset(80)
    loader = DataLoader(dataset=dataset, batch_size=4)
    loss = Loss(mse, name='MSE')
    learning_schema = LearningSchema.sgd(momentum=0)
    return Trainer(
        model,
        name='MyDDPTrainer',
        loader=loader,
        learning_schema=learning_schema,
        loss=loss,
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


@pytest.mark.skipif(sys.platform != 'linux', reason='ddp only works on linux')
@pytest.mark.parametrize('world_size', [WORLD_SIZE])
def test_ddp_convergence(example_run_id, tmp_path, world_size) -> None:
    """Test convergence of DDP training."""
    running_worker = RunningWorker(
        assert_convergence, par_dir=tmp_path, run_id=example_run_id
    )
    worker = DistributedWorker(running_worker, world_size=world_size)
    exit_codes, _ = worker.process()
    assert all(exit_code == 0 for exit_code in exit_codes)

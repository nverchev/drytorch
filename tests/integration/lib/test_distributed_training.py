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
from drytorch.core import exceptions
from drytorch.core import protocols as p


WORLD_SIZE = 2


def mse(outputs: TorchData, targets: torch.Tensor) -> torch.Tensor:
    """Mean square error calculation from structured outputs."""
    return ((outputs.output - targets) ** 2).mean()


class GradRecord(p.GradientOpProtocol):
    """Gradient operation that tracks gradients."""

    def __init__(self, grad_list: list[torch.Tensor]) -> None:
        """Constructor."""
        self.grad_list = grad_list

    def __call__(self, params: list[torch.Tensor]) -> None:
        """Record gradients."""
        for param in params:
            if param.grad is not None:
                self.grad_list.append(param.grad)

        return


def setup_training() -> Trainer[TorchTuple, torch.Tensor, TorchData]:
    """Setup DDP training."""
    network = Linear(1, 1)
    model = Model(network, name='linear')
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


def setup_training_with_no_ddp_module() -> Trainer[
    TorchTuple, torch.Tensor, TorchData
]:
    """Setup DDP training without DDP module."""
    model = Model(Linear(1, 1), name='linear', should_distribute=False)
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


def assert_ddp_warning_is_raised() -> None:
    """Assert that a warning is raised when the module is not DDP."""
    trainer = setup_training_with_no_ddp_module()
    with pytest.warns(exceptions.ModuleNotDistributedWarning):
        trainer.train(1)

    return


def gather_ddp_outputs() -> list[TorchData]:
    """Worker function to test distributed output gathering."""
    trainer = setup_training()

    # different outputs for each rank
    rank = torch.distributed.get_rank()
    dummy_output = 2 * [TorchData(torch.tensor([rank]))]
    trainer.outputs_list.extend(dummy_output)

    trainer._gather_stored_outputs()

    return trainer.outputs_list


def get_trained_metric() -> float:
    """Get the final loss value."""
    trainer = setup_training()
    trainer.train(1)
    return trainer.computed_metrics[trainer.objective.name]


def get_grads_and_params() -> tuple[list[nn.Parameter], list[torch.Tensor]]:
    """Get model gradients and parameters."""
    trainer = setup_training()
    gradients = []
    trainer._model_optimizer.gradient_op = GradRecord(gradients)
    trainer.train(1)
    return list(trainer.model.module.parameters()), gradients


def save_and_load() -> list[nn.Parameter]:
    """Save and load model parameters."""
    trainer = setup_training()
    trainer.save_checkpoint()
    trainer.load_checkpoint()
    return list(trainer.model.module.parameters())


@pytest.mark.skipif(sys.platform != 'linux', reason='ddp only works on linux')
@pytest.mark.parametrize('world_size', [WORLD_SIZE])
def test_ddp_warning(example_run_id, tmp_path, world_size) -> None:
    """Test that missing ddp in module triggers warning."""
    running_worker = RunningWorker(
        assert_ddp_warning_is_raised, par_dir=tmp_path, run_id=example_run_id
    )
    worker = DistributedWorker(running_worker, world_size=world_size)
    exit_codes, _ = worker.process()

    assert all(exit_code == 0 for exit_code in exit_codes)


@pytest.mark.skipif(sys.platform != 'linux', reason='ddp only works on linux')
@pytest.mark.parametrize('world_size', [WORLD_SIZE])
def test_ddp_synchronization(example_run_id, tmp_path, world_size) -> None:
    """Test that gradients are synchronized across ranks."""
    running_worker = RunningWorker(
        get_grads_and_params, par_dir=tmp_path, run_id=example_run_id
    )
    worker = DistributedWorker(running_worker, world_size=world_size)
    exit_codes, return_dict = worker.process()

    assert all(exit_code == 0 for exit_code in exit_codes)
    for rank in range(1, world_size):
        for param0, param_rank in zip(
            return_dict[0][0], return_dict[rank][0], strict=True
        ):
            assert torch.allclose(param0, param_rank)

        for grad0, grad_rank in zip(
            return_dict[0][1], return_dict[rank][1], strict=True
        ):
            assert torch.allclose(grad0, grad_rank)


@pytest.mark.skipif(sys.platform != 'linux', reason='ddp only works on linux')
@pytest.mark.parametrize('world_size', [WORLD_SIZE])
def test_metrics_are_averaged(example_run_id, tmp_path, world_size) -> None:
    """Test that metrics are averaged across ranks."""
    running_worker = RunningWorker(
        get_trained_metric, par_dir=tmp_path, run_id=example_run_id
    )
    worker = DistributedWorker(running_worker, world_size=world_size)
    exit_codes, return_dict = worker.process()

    assert all(exit_code == 0 for exit_code in exit_codes)
    for rank in range(1, world_size):
        assert return_dict[rank] == return_dict[rank]


@pytest.mark.skipif(sys.platform != 'linux', reason='ddp only works on linux')
@pytest.mark.parametrize('world_size', [WORLD_SIZE])
def test_checkpointing(example_run_id, tmp_path, world_size) -> None:
    """Test checkpointing."""
    running_worker = RunningWorker(
        save_and_load, par_dir=tmp_path, run_id=example_run_id
    )
    worker = DistributedWorker(running_worker, world_size=world_size)
    exit_codes, return_dict = worker.process()

    assert all(exit_code == 0 for exit_code in exit_codes)
    for rank in range(1, world_size):
        for param0, param_rank in zip(
            return_dict[0][0], return_dict[rank][0], strict=True
        ):
            assert torch.allclose(param0, param_rank)


@pytest.mark.skipif(sys.platform != 'linux', reason='ddp only works on linux')
@pytest.mark.parametrize('world_size', [WORLD_SIZE])
def test_distributed_gather_outputs(
    example_run_id, tmp_path, world_size
) -> None:
    """Test that outputs from all ranks are gathered to rank 0."""
    running_worker = RunningWorker(
        gather_ddp_outputs, par_dir=tmp_path, run_id=example_run_id
    )
    worker = DistributedWorker(running_worker, world_size=world_size)
    exit_codes, return_dict = worker.process()

    assert all(exit_code == 0 for exit_code in exit_codes)

    # rank 0 should have gathered outputs from all ranks (size = world_size)
    rank_0_outputs = return_dict[0]
    assert len(rank_0_outputs) == 2 * world_size

    # verify content: should contain 0s from rank 0 and 1s from rank 1
    gathered_values = [t.output.item() for t in rank_0_outputs]
    for r in range(world_size):
        assert r in gathered_values

    # other ranks should have cleared their lists
    for rank in range(1, world_size):
        assert len(return_dict[rank]) == 0

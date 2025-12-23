"""Configuration module with objects from the package."""

import os
import pathlib
import socket
import uuid
import warnings

from collections.abc import Callable, Generator
from multiprocessing import managers
from typing import Generic, ParamSpec, TypeVar

import torch
import torch.multiprocessing as mp

from torch import distributed as dist

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
from drytorch.core.exceptions import ExperimentalFeatureWarning
from drytorch.core.experiment import Run
from tests.simple_classes import IdentityDataset, Linear, TorchData, TorchTuple


T = TypeVar('T')
P = ParamSpec('P')


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


class DistributedWorker(Generic[P, T]):
    """Callable wrapper to distribute a worker across multiple processes."""

    def __init__(self, worker: Callable[P, T], world_size: int) -> None:
        """Constructor."""
        self.worker = worker
        self.world_size = world_size
        self.port = self._get_free_port()
        return

    def __call__(
        self, rank: int, return_dict: managers.DictProxy, *args: P.args
    ) -> None:
        """Run the worker in a distributed environment."""
        self._setup_distributed(rank)
        try:
            return_dict[rank] = self.worker(*args)
        finally:
            self._cleanup_distributed()

        return

    def process(self, *args: P.args) -> tuple[list[int], dict[int, T]]:
        """Run processes with multiprocessing.Manager."""
        ctx = mp.get_context('spawn')
        manager = ctx.Manager()
        return_dict = manager.dict()

        processes = []
        for rank in range(self.world_size):
            p = ctx.Process(target=self, args=(rank, return_dict, *args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return [p.exitcode for p in processes], dict(return_dict)

    def _setup_distributed(self, rank: int) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.port
        dist.init_process_group(
            backend='gloo',
            rank=rank,
            init_method='tcp://127.0.0.1:29500',
            world_size=self.world_size,
        )
        return

    @staticmethod
    def _get_free_port() -> str:
        """Find an available port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return str(s.getsockname()[1])

    @staticmethod
    def _cleanup_distributed() -> None:
        """Clean up the distributed environment."""
        dist.destroy_process_group()
        return


class RunningWorker(Generic[P, T]):
    """Callable wrapper for multiprocessing workers that runs the experiment."""

    def __init__(
        self,
        worker: Callable[P, T],
        par_dir: pathlib.Path,
        run_id: str,
        name: str | None = None,
        should_remove_default_tracking: bool = True,
    ) -> None:
        """Constructor."""
        self.worker = worker
        self.par_dir = par_dir
        self.run_id = run_id
        if name is None:  # avoid conflicts between parallel tests
            name = f'TestSubProcessExperiment_{uuid.uuid4()}'
        self.name = name
        self.should_remove_default_tracking = should_remove_default_tracking
        return

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call the worker with the experiment set up."""
        if self.should_remove_default_tracking:
            drytorch.remove_all_default_trackers()
        exp: Experiment = Experiment(
            name=self.name, par_dir=self.par_dir, config=None
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ExperimentalFeatureWarning)
            with exp.create_run(run_id=self.run_id):
                return self.worker(*args, **kwargs)

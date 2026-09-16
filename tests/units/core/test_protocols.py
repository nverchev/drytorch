"""Tests that plain Python objects can match the protocols.

basedpyright checks the assignments below: one fails when a class does not
match the protocol, so each class states the minimum a protocol asks for.
The tests check the protocols that are runtime-checkable.
"""

import datetime

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

from torch.utils import data

import pytest

from drytorch.core import protocols as p


class _Checkpoint:
    """Minimal checkpoint."""

    def bind_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Bind the model to manage."""

    def bind_module(self, name: str, module: torch.nn.Module) -> None:
        """Bind a module connected to the model."""

    def bind_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Bind the optimizer connected to the model."""

    def load(self, epoch: int = -1) -> None:
        """Load the model and optimizer state dictionaries."""

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""


class _Module:
    """Minimal module."""

    def forward(self, inputs: torch.Tensor, /) -> torch.Tensor:
        """Forward run of the network."""
        return inputs


class _Model:
    """Minimal model."""

    def __init__(self, module: torch.nn.Module) -> None:
        """Initialize."""
        self.checkpoint: p.CheckpointProtocol = _Checkpoint()
        self.device = torch.device('cpu')
        self.epoch = 0
        self.mixed_precision = False
        self.module = module
        self.name = 'model'

    def __call__(self, inputs: Any) -> Any:
        """Call the module forward method."""
        return self.module(inputs)

    def increment_epoch(self) -> None:
        """Increment the epoch by 1."""
        self.epoch += 1

    def post_batch_update(self) -> None:
        """Update the model after processing a batch of data."""

    def post_epoch_update(self) -> None:
        """Update the model after processing an epoch of data."""


class _Loader:
    """Minimal loader."""

    def __init__(self, dataset: data.Dataset[Any]) -> None:
        """Initialize."""
        self.batch_size = 1
        self.dataset = dataset
        self.sampler: data.Sampler[Any] | Iterable[Any] = []

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Return an iterator over the dataset in batches."""
        return iter(())

    def __len__(self) -> int:
        """Return the number of batches."""
        return 0


class _Objective:
    """Minimal objective."""

    def compute(self) -> Mapping[str, torch.Tensor]:
        """Return the computed values by name."""
        return {}

    def reset(self) -> None:
        """Reset cached values."""

    def sync(self) -> None:
        """Synchronize the state across processes."""

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, /) -> None:
        """Compute the metrics only."""


class _Loss(_Objective):
    """Minimal loss."""

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, /
    ) -> torch.Tensor:
        """Process the outputs and targets and return the loss."""
        return torch.zeros(1)


class _Scheduler:
    """Minimal scheduler."""

    def __call__(self, base_lr: float, epoch: int) -> float:
        """Modify the learning rate according to a schedule."""
        return base_lr


class _GradientOp:
    """Minimal gradient operation."""

    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Apply the gradient operation to the given parameters."""


class _LearningSchema:
    """Minimal learning schema."""

    def __init__(self) -> None:
        """Initialize."""
        self.base_lr: float | Mapping[str, float] = 0.1
        self.gradient_op: p.GradientOpProtocol = _GradientOp()
        self.optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD
        self.optimizer_defaults: Mapping[str, Any] = {}
        self.scheduler: p.SchedulerProtocol = _Scheduler()


class _Monitor:
    """Minimal monitor of a model."""

    def __init__(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Initialize."""
        self.computed_metrics: Mapping[str, float] = {}
        self.created_at = datetime.datetime.now()
        self.higher_is_better: bool | None = None
        self.model = model
        self.name = 'monitor'
        self.objective: p.ObjectiveProtocol[Any, Any] = _Objective()

    def __call__(self, store_outputs: bool = False) -> None:
        """Run the model on the data once."""


_actor: type[p.ActorProtocol] = _Monitor
_checkpoint: type[p.CheckpointProtocol] = _Checkpoint
_created_at: type[p.CreatedAtProtocol] = _Monitor
_gradient_op: type[p.GradientOpProtocol] = _GradientOp
_higher_is_better: type[p.HigherIsBetterProtocol] = _Monitor
_learning: type[p.LearningProtocol] = _LearningSchema
_loader: type[p.LoaderProtocol[tuple[torch.Tensor, torch.Tensor]]] = _Loader
_loss: type[p.LossProtocol[torch.Tensor, torch.Tensor]] = _Loss
_model: type[p.ModelProtocol[Any, Any]] = _Model
_module: type[p.ModuleProtocol[torch.Tensor, torch.Tensor]] = _Module
_monitor: type[p.MonitorProtocol] = _Monitor
_named: type[p.NamedProtocol] = _Monitor
_objective: type[p.ObjectiveProtocol[torch.Tensor, torch.Tensor]] = _Objective
_objective_monitor: type[p.ObjectiveMonitorProtocol] = _Monitor
_scheduler: type[p.SchedulerProtocol] = _Scheduler
_sync: type[p.SyncProtocol] = _Objective


@pytest.fixture
def loss() -> _Loss:
    """A minimal loss."""
    return _Loss()


@pytest.fixture
def model() -> _Model:
    """A minimal model."""
    return _Model(torch.nn.Linear(1, 1))


@pytest.fixture
def monitor(model) -> _Monitor:
    """A minimal monitor of a model."""
    return _Monitor(model)


@pytest.fixture
def objective() -> _Objective:
    """A minimal objective."""
    return _Objective()


@pytest.mark.parametrize(
    ('fixture_name', 'protocol'),
    [
        ('loss', p.LossProtocol),
        ('loss', p.ObjectiveProtocol),
        ('model', p.ModelProtocol),
        ('model', p.NamedProtocol),
        ('monitor', p.ActorProtocol),
        ('monitor', p.CreatedAtProtocol),
        ('monitor', p.HigherIsBetterProtocol),
        ('monitor', p.NamedProtocol),
        ('monitor', p.ObjectiveMonitorProtocol),
        ('objective', p.ObjectiveProtocol),
        ('objective', p.SyncProtocol),
    ],
)
def test_object_matches_runtime_protocol(
    request, fixture_name, protocol
) -> None:
    """Test the object satisfies the runtime-checkable protocol."""
    instance = request.getfixturevalue(fixture_name)

    assert isinstance(instance, protocol)

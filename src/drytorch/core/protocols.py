"""Internal protocols."""

from __future__ import annotations

import abc
import datetime

from collections.abc import Iterable, Iterator, Mapping, MutableSequence
from typing import (
    Any,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch

from torch.utils import data


__all__ = [
    'CheckpointProtocol',
    'CreatedAtProtocol',
    'GradientOpProtocol',
    'HigherIsBetterProtocol',
    'HookProtocol',
    'InputType',
    'LearningProtocol',
    'LoaderProtocol',
    'LossProtocol',
    'ModelProtocol',
    'ModuleProtocol',
    'NamedProtocol',
    'ObjectiveProtocol',
    'OutputType',
    'RunnerProtocol',
    'SchedulerProtocol',
    'SourceProtocol',
    'SyncProtocol',
    'TargetType',
    'Tensors',
    'TrainerProtocol',
]

# pyright: reportReturnType=false

_T = TypeVar('_T')
Tensors: TypeAlias = torch.Tensor | MutableSequence[torch.Tensor]
InputType: TypeAlias = Union[Tensors, NamedTuple]
OutputType: TypeAlias = Any
TargetType: TypeAlias = Union[Tensors, NamedTuple]

_Data_co = TypeVar(
    '_Data_co', bound=tuple[InputType, TargetType], covariant=True
)
_Output_co = TypeVar('_Output_co', bound=OutputType, covariant=True)
_Target_co = TypeVar('_Target_co', bound=TargetType, covariant=True)

_Input_contra = TypeVar('_Input_contra', bound=InputType, contravariant=True)
_Target_contra = TypeVar('_Target_contra', bound=TargetType, contravariant=True)
_Output_contra = TypeVar('_Output_contra', bound=OutputType, contravariant=True)


# ---------------------------------------------------------------------------
# Component protocols
# ---------------------------------------------------------------------------


class CheckpointProtocol(Protocol):
    """Protocol that stores and loads weight for a ModelProtocol class.

    Implementations check that the model belongs to the current run
    (:func:`~drytorch.core.registering.check_current_run`) and emit
    :class:`~drytorch.core.log_events.SaveModelEvent` on ``save`` and
    :class:`~drytorch.core.log_events.LoadModelEvent` on ``load``.
    """

    def bind_model(self, model: ModelProtocol[Any, Any]) -> None:
        """Bind the model to manage."""

    def bind_module(self, name: str, module: torch.nn.Module) -> None:
        """Bind a module connected to the model."""

    def bind_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Bind the optimizer connected to the model."""

    def load(self, epoch: int = -1) -> int:
        """Load the model and optimizer state dictionaries.

        :meth:`ModelProtocol.load_state` also restores the epoch of the model.

        Args:
            epoch: the epoch to load. Default loads the last saved epoch.

        Returns:
            The loaded epoch.
        """

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""


class GradientOpProtocol(Protocol):
    """Abstract base class for gradient operations."""

    @abc.abstractmethod
    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Apply the gradient operation to the given parameters."""


class HookProtocol(Protocol):
    """Protocol for an object that a trainer calls during training."""

    @property
    @abc.abstractmethod
    def last_run_epoch(self) -> int:
        """The epoch when the hook last ran, or 0 if never run."""

    @property
    @abc.abstractmethod
    def trainer(self) -> TrainerProtocol[Any, Any, Any]:
        """The trainer that calls the hook."""

    def __call__(self) -> None:
        """Execute the hook."""

    def bind_trainer(self, trainer: TrainerProtocol[Any, Any, Any]) -> None:
        """Bind the hook to the trainer that calls it.

        Args:
            trainer: the trainer calling the hook.
        """


class LearningProtocol(Protocol):
    """Protocol with specifications for the learning algorithm."""

    @property
    @abc.abstractmethod
    def base_lr(self) -> float | Mapping[str, float]:
        """Initial learning rates for named parameters or global value."""

    @property
    @abc.abstractmethod
    def gradient_op(self) -> GradientOpProtocol:
        """Operations on the model gradients."""

    @property
    @abc.abstractmethod
    def optimizer_cls(self) -> type[torch.optim.Optimizer]:
        """The optimizer class to bind to the module."""

    @property
    @abc.abstractmethod
    def optimizer_defaults(self) -> Mapping[str, Any]:
        """Optional arguments for the optimizer."""

    @property
    @abc.abstractmethod
    def scheduler(self) -> SchedulerProtocol:
        """Modifies the learning rate given the current epoch."""


class LoaderProtocol(Protocol[_Data_co]):
    """Protocol loading and batching a dataset."""

    @property
    @abc.abstractmethod
    def batch_size(self) -> int | None:
        """The batch size."""

    @property
    @abc.abstractmethod
    def dataset(self) -> data.Dataset[Any]:
        """The dataset to load."""

    @property
    @abc.abstractmethod
    def sampler(self) -> torch.utils.data.Sampler[Any] | Iterable[Any]:
        """The sampler used to select the samples."""

    def __iter__(self) -> Iterator[_Data_co]:
        """Return an iterator over the dataset in batches."""

    def __len__(self) -> int:
        """Return the global number of batches in the dataset across all ranks.

        In distributed training, this is the total batch count across the entire
        world size, not just the number of batches yielded by the local process.
        """


@runtime_checkable
class ModelProtocol(Protocol[_Input_contra, _Output_co]):
    """Protocol for a wrapper around a torch module.

    Implementations register with
    :func:`~drytorch.core.registering.register_model` when created, which
    emits :class:`~drytorch.core.log_events.ModelRegistrationEvent`;
    ``increment_epoch`` emits
    :class:`~drytorch.core.log_events.IncrementEpochEvent`.

    """

    @abc.abstractmethod
    def __call__(self, /, inputs: _Input_contra) -> _Output_co:
        """Call the module forward method."""

    @property
    @abc.abstractmethod
    def epoch(self) -> int:
        """The number of epochs the model has been trained so far."""

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""

    @property
    @abc.abstractmethod
    def mixed_precision(self) -> bool:
        """Whether to use mixed precision computing."""

    @property
    def module(self) -> torch.nn.Module:
        """The module wrapped by the class."""

    @property
    def name(self) -> str:
        """The name of the model."""

    @abc.abstractmethod
    def bind_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Bind the optimizer whose state is saved with the model.

        Args:
            optimizer: the optimizer updating the model parameters.
        """

    @abc.abstractmethod
    def increment_epoch(self) -> None:
        """Increment the epoch by 1."""

    @abc.abstractmethod
    def load_state(self, epoch: int = -1) -> None:
        """Load the weights, the optimizer state and the epoch.

        Args:
            epoch: the epoch to load. Default loads the last saved epoch.
        """

    @abc.abstractmethod
    def post_batch_update(self) -> None:
        """Update the model after processing a batch of data."""

    @abc.abstractmethod
    def post_epoch_update(self) -> None:
        """Update the model after processing an epoch of data."""

    @abc.abstractmethod
    def save_state(self) -> None:
        """Save the weights, the optimizer state and the epoch."""


@runtime_checkable
class ObjectiveProtocol(Protocol[_Output_contra, _Target_contra]):
    """Protocol that calculates and returns metrics."""

    @abc.abstractmethod
    def update(
        self, outputs: _Output_contra, targets: _Target_contra, /
    ) -> Any:
        """Compute the metrics only.

        Args:
            outputs: model outputs.
            targets: ground truth.
        """

    @abc.abstractmethod
    def compute(self) -> Mapping[str, torch.Tensor] | torch.Tensor | None:
        """Return a mapping from the metric names to the calculated values."""

    @abc.abstractmethod
    def reset(self) -> Any:
        """Reset cached values."""


@runtime_checkable
class LossProtocol(ObjectiveProtocol[_Output_contra, _Target_contra], Protocol):
    """Protocol that calculates and returns metrics and the loss."""

    def forward(
        self, outputs: _Output_contra, targets: _Target_contra, /
    ) -> torch.Tensor:
        """Process the outputs and targets and returns the loss.

        Args:
            outputs: model outputs.
            targets: ground truth.

        Returns:
            The computed loss.
        """


class SchedulerProtocol(Protocol):
    """Protocol of a scheduler for the learning rate."""

    def __call__(self, base_lr: float, epoch: int) -> float:
        """Modify the learning rate according to a schedule.

        Args:
            base_lr: initial learning rate.
            epoch: the current epoch.

        Returns:
            The scheduled value for the learning rate.
        """


@runtime_checkable
class RunnerProtocol(Protocol):
    """Protocol for an object that runs a model and is recorded with the run.

    Implementations register with
    :func:`~drytorch.core.registering.register_runner`, which emits
    :class:`~drytorch.core.log_events.RunnerRegistrationEvent`.
    """

    @property
    @abc.abstractmethod
    def model(self) -> ModelProtocol[Any, Any]:
        """The model that the runner runs."""


class SourceProtocol(RunnerProtocol, Protocol):
    """Protocol for a class that runs a model and reports its metrics.

    Implementations register as runners of their model with
    :func:`~drytorch.core.registering.register_runner`, which emits
    :class:`~drytorch.core.log_events.RunnerRegistrationEvent`. Each pass
    over the data emits :class:`~drytorch.core.log_events.IterateBatchEvent`
    and then :class:`~drytorch.core.log_events.MetricEvent`.
    """

    def __call__(self, store_outputs: bool = False) -> None:
        """Run the model on the data once.

        Args:
            store_outputs: whether to store the model outputs.
        """

    @property
    def computed_metrics(self) -> Mapping[str, float]:
        """The metrics computed at the current epoch, or empty if not run."""

    @property
    def last_run_epoch(self) -> int:
        """The epoch when the source last ran, or 0 if never run."""

    @property
    def name(self) -> str:
        """The name of the source."""

    @property
    @abc.abstractmethod
    def objective(self) -> ObjectiveProtocol[Any, Any]:
        """The objective computing the metrics."""


@runtime_checkable
class TrainerProtocol(
    SourceProtocol,
    Protocol[_Input_contra, _Target_co, _Output_co],
):
    """Protocol for a class that train a model.

    Besides the duties of :class:`SourceProtocol`, ``train`` emits
    :class:`~drytorch.core.log_events.StartTrainingEvent`, then
    :class:`~drytorch.core.log_events.StartEpochEvent` and
    :class:`~drytorch.core.log_events.EndEpochEvent` for each epoch, and
    :class:`~drytorch.core.log_events.EndTrainingEvent` unless terminated;
    ``terminate_training`` emits
    :class:`~drytorch.core.log_events.TerminatedTrainingEvent` and
    ``update_learning_rate`` emits
    :class:`~drytorch.core.log_events.LearningRateEvent`.

    """

    @property
    @abc.abstractmethod
    def learning_schema(self) -> LearningProtocol:
        """Optimizer settings and scheduling given at creation."""

    @property
    @abc.abstractmethod
    def model(self) -> ModelProtocol[_Input_contra, _Output_co]:
        """The model to train."""

    @property
    def terminated(self) -> bool:
        """If true, this trainer should not be used for training anymore."""

    def train(self, n_epochs: int) -> None:
        """Train the module for the specified number of epochs.

        Args:
            n_epochs: the number of epochs for which train the module.
        """

    def terminate_training(self, reason: str) -> None:
        """Prevent the trainer from continue the training."""

    def save_checkpoint(self) -> None:
        """Save model and optimizer state in a checkpoint."""

    def load_checkpoint(self, epoch: int = -1) -> None:
        """Load model and optimizer state from a checkpoint."""

    def update_learning_rate(
        self,
        base_lr: float | Mapping[str, float] | None = None,
        scheduler: SchedulerProtocol | None = None,
    ) -> None:
        """Update the learning rate(s).

        Args:
            base_lr: initial learning rates for named parameters or global
                value. Default keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.
        """


# ---------------------------------------------------------------------------
# Contract protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CreatedAtProtocol(Protocol):
    """Protocol for objects exposing a creation timestamp."""

    @property
    @abc.abstractmethod
    def created_at(self) -> datetime.datetime:
        """The creation timestamp of the object."""


@runtime_checkable
class HigherIsBetterProtocol(Protocol):
    """Protocol for metrics exposing whether higher values are better."""

    @property
    @abc.abstractmethod
    def higher_is_better(self) -> bool | None:
        """Flag whether higher metric values are better."""


class ModuleProtocol(Protocol[_Input_contra, _Output_co]):
    """Protocol for a PyTorch module with type annotations."""

    @abc.abstractmethod
    def forward(self, inputs: _Input_contra, /) -> _Output_co:
        """Forward run of the network."""


@runtime_checkable
class NamedProtocol(Protocol):
    """Protocol for objects exposing a name."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the object."""


@runtime_checkable
class SyncProtocol(Protocol):
    """Protocol for objects that synchronize their state across processes."""

    def sync(self) -> None:
        """Synchronize the state across processes."""

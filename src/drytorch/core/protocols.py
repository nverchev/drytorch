"""Module containing internal protocols."""

from __future__ import annotations

import abc

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
    'GradientOpProtocol',
    'InputType',
    'LearningProtocol',
    'LoaderProtocol',
    'LossProtocol',
    'ModelProtocol',
    'MonitorProtocol',
    'ObjectiveProtocol',
    'OutputType',
    'SchedulerProtocol',
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

_Input_contra = TypeVar('_Input_contra', bound=InputType, contravariant=True)
_Target_contra = TypeVar('_Target_contra', bound=TargetType, contravariant=True)
_Output_contra = TypeVar('_Output_contra', bound=OutputType, contravariant=True)

_Input = TypeVar('_Input', bound=InputType)
_Target = TypeVar('_Target', bound=TargetType)
_Output = TypeVar('_Output', bound=OutputType)


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
    emits :class:`~drytorch.core.log_events.ModelRegistrationEvent`.

    Attributes:
        epoch: the number of epochs the model has been trained so far.
    """

    epoch: int

    @property
    @abc.abstractmethod
    def checkpoint(self) -> CheckpointProtocol:
        """The object responsible for saving and loading the model."""

    @property
    @abc.abstractmethod
    def mixed_precision(self) -> bool:
        """Whether to use mixed precision computing."""

    @abc.abstractmethod
    def __call__(self, /, inputs: _Input_contra) -> _Output_co:
        """Call the module forward method."""

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""

    @property
    def module(self) -> torch.nn.Module:
        """The module wrapped by the class."""

    @property
    def name(self) -> str:
        """The name of the model."""

    @abc.abstractmethod
    def increment_epoch(self) -> None:
        """Increment the epoch by 1."""

    @abc.abstractmethod
    def post_batch_update(self) -> None:
        """Update the model after processing a batch of data."""

    @abc.abstractmethod
    def post_epoch_update(self) -> None:
        """Update the model after processing an epoch of data."""


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

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""

    def load(self, epoch: int = -1) -> None:
        """Load the model and optimizer state dictionaries."""


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


class GradientOpProtocol(Protocol):
    """Abstract base class for gradient operations."""

    @abc.abstractmethod
    def __call__(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Apply the gradient operation to the given parameters."""


class LearningProtocol(Protocol):
    """Protocol with specifications for the learning algorithm.

    Attributes:
        optimizer_cls: the optimizer class to bind to the module.
        base_lr: initial learning rates for named parameters or global value.
        optimizer_defaults: optional arguments for the optimizer.
        scheduler: modifies the learning rate given the current epoch.
    """

    optimizer_cls: type[torch.optim.Optimizer]
    base_lr: float | Mapping[str, float]
    scheduler: SchedulerProtocol
    optimizer_defaults: Mapping[str, Any]
    gradient_op: GradientOpProtocol


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


class MonitorProtocol(Protocol):
    """Protocol for a class that validates a model.

    Implementations register as actors of their model with
    :func:`~drytorch.core.registering.register_actor`, which emits
    :class:`~drytorch.core.log_events.ActorRegistrationEvent`. Each pass
    over the data emits :class:`~drytorch.core.log_events.IterateBatchEvent`
    and then :class:`~drytorch.core.log_events.MetricEvent`.

    Attributes:
        model: the model to evaluate.
    """

    model: ModelProtocol[Any, Any]

    def __call__(self, store_outputs: bool = False) -> None:
        """Run the model on the data once.

        Args:
            store_outputs: whether to store the model outputs.
        """

    @property
    def name(self) -> str:
        """The name of the monitor."""

    @property
    def computed_metrics(self) -> Mapping[str, float]:
        """Computed metric values."""


@runtime_checkable
class TrainerProtocol(
    MonitorProtocol,
    Protocol[_Input, _Target, _Output],
):
    """Protocol for a class that train a model.

    Besides the duties of :class:`MonitorProtocol`, ``train`` emits
    :class:`~drytorch.core.log_events.StartTrainingEvent`, then
    :class:`~drytorch.core.log_events.StartEpochEvent` and
    :class:`~drytorch.core.log_events.EndEpochEvent` for each epoch, and
    :class:`~drytorch.core.log_events.EndTrainingEvent` unless terminated;
    ``terminate_training`` emits
    :class:`~drytorch.core.log_events.TerminatedTrainingEvent` and
    ``update_learning_rate`` emits
    :class:`~drytorch.core.log_events.LearningRateEvent`.

    Attributes:
        model: the model to train.
        objective: determines the optimization's criterion.
    """

    model: ModelProtocol[_Input, _Output]
    objective: LossProtocol[_Output, _Target]

    @property
    @abc.abstractmethod
    def learning_schema(self) -> LearningProtocol:
        """Optimizer settings and scheduling."""

    @property
    def terminated(self) -> bool:
        """If true, this trainer should not be used for training anymore."""

    @property
    @abc.abstractmethod
    def validation(self) -> MonitorProtocol | None:
        """The validation run during training, if any."""

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
        base_lr: float | None,
        scheduler: SchedulerProtocol | None,
    ) -> None:
        """Update the learning rate(s).

        It updates the learning rates for each parameter's group in the
        optimizer based on input learning rate and scheduler.

        Args:
            base_lr: the initial learning rate.
            scheduler: scheduler for the learning rate.
        """

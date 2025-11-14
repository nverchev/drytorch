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


# pyright: reportReturnType=false

_T = TypeVar('_T')
Tensors: TypeAlias = torch.Tensor | MutableSequence[torch.Tensor]
InputType: TypeAlias = Union[Tensors, NamedTuple]  # noqa: UP007
OutputType: TypeAlias = Any
TargetType: TypeAlias = Union[Tensors, NamedTuple]  # noqa: UP007

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
    """Protocol loading and batching a dataset.

    Attributes:
        batch_size: the batch size.
    """

    batch_size: int | None

    def get_dataset(self) -> data.Dataset[_Data_co]:
        """Returns the dataset."""

    def __iter__(self) -> Iterator[_Data_co]:
        """Return an iterator over the dataset in batches."""

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""


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


class ModuleProtocol(Protocol[_Input_contra, _Output_co]):
    """Protocol for a PyTorch module with type annotations."""

    def forward(self, inputs: _Input_contra, /) -> _Output_co:
        """Forward run of the network."""


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
    base_lr: float | dict[str, float]
    scheduler: SchedulerProtocol
    optimizer_defaults: dict[str, Any]
    gradient_op: GradientOpProtocol | None


@runtime_checkable
class ModelProtocol(Protocol[_Input_contra, _Output_co]):
    """Protocol for a wrapper around a torch module.

    Attributes:
        module: Pytorch module to optimize.
        epoch: the number of epochs the model has been trained so far.
        checkpoint: the object responsible for saving and loading the model.
        mixed_precision: whether to use mixed precision computing.
    """

    module: torch.nn.Module
    epoch: int
    checkpoint: CheckpointProtocol
    mixed_precision: bool

    @abc.abstractmethod
    def __call__(self, /, inputs: _Input_contra) -> _Output_co:
        """Call the module forward method."""

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""

    @property
    def name(self) -> str:
        """The name of the model."""

    @abc.abstractmethod
    def increment_epoch(self):
        """Increment the epoch by 1."""

    def update_parameters(self) -> None:
        """Update the parameters of the model."""


class CheckpointProtocol(Protocol):
    """Protocol that stores and loads weight for a ModelProtocol class."""

    def bind_model(self, model: ModelProtocol[Any, Any]):
        """Bind the model to manage."""

    def bind_optimizer(self, optimizer: torch.optim.Optimizer):
        """Bind the optimizer connected to the model."""

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""

    def load(self, epoch: int = -1) -> None:
        """Load the model and optimizer state dictionaries."""


class MonitorProtocol(Protocol):
    """Protocol for a class that validates a model.

    Attributes:
        model: the model to evaluate.
    """

    model: ModelProtocol[Any, Any]

    @property
    def name(self) -> str:
        """The name of the model."""

    @property
    def computed_metrics(self) -> Mapping[str, float]:
        """Computed metric values."""


@runtime_checkable
class TrainerProtocol(
    MonitorProtocol,
    Protocol[_Input, _Target, _Output],
):
    """Protocol for a class that train a model.

    Attributes:
        model: the model to train.
    """

    model: ModelProtocol[_Input, _Output]
    learning_schema: LearningProtocol
    validation: MonitorProtocol | None
    objective: LossProtocol[_Output, _Target]

    @property
    def terminated(self) -> bool:
        """If true, this trainer should not be used for training anymore."""

    def train(self, num_epochs: int) -> None:
        """Train the module for the specified number of epochs.

        Args:
            num_epochs: the number of epochs for which train the module.
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

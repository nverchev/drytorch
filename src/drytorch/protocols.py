"""Module containing internal protocols."""

from __future__ import annotations

import abc
from collections.abc import Iterator, Iterable, Mapping, MutableSequence
from typing import Any, Optional, Protocol, SupportsIndex, TypeAlias, TypeVar
from typing import Callable, Union
from typing import runtime_checkable

import torch
from torch.utils import data

_T = TypeVar('_T')

Tensors: TypeAlias = Union[torch.Tensor, MutableSequence[torch.Tensor]]


@runtime_checkable
class NamedTupleProtocol(Protocol[_T]):
    """
    Optional protocol for the input and target types.

    Correctly handled by the default collate function.
    NamedTuples containing different types are currently interpreted as
    Generic[Any]. At the moment, this protocol won't support these interfaces.
    """
    _fields: tuple

    def __getitem__(self, index: SupportsIndex) -> _T:
        ...

    def __len__(self) -> int:
        ...

    def _asdict(self) -> dict[str, _T]:
        ...


InputType: TypeAlias = Tensors | NamedTupleProtocol[Tensors]
OutputType: TypeAlias = Any
TargetType: TypeAlias = Tensors | NamedTupleProtocol[Tensors]

_Data_co = TypeVar('_Data_co',
                   bound=tuple[InputType, TargetType],
                   covariant=True)
_Output_co = TypeVar('_Output_co',
                     bound=OutputType,
                     covariant=True)

_Input_contra = TypeVar('_Input_contra',
                        bound=InputType,
                        contravariant=True)
_Target_contra = TypeVar('_Target_contra',
                         bound=TargetType,
                         contravariant=True)
_Output_contra = TypeVar('_Output_contra',
                         bound=OutputType,
                         contravariant=True)

_Input = TypeVar('_Input', bound=InputType)
_Target = TypeVar('_Target', bound=TargetType)
_Output = TypeVar('_Output', bound=OutputType)


class LoaderProtocol(Protocol[_Data_co]):
    """
    Protocol loading and batching a dataset.

    Attributes:
        batch_size: the batch size.
        dataset: dataset
    """
    batch_size: Optional[int]
    dataset: data.Dataset

    def __iter__(self) -> Iterator[_Data_co]:
        """Return an iterator over the dataset in batches"""

    def __len__(self) -> int:
        """Return the number of batches in the dataset"""


class SchedulerProtocol(Protocol):
    """
    Protocol of a scheduler compatible with the LearningScheme class.
    """

    def __call__(self, base_lr: float, epoch: int) -> float:
        """
        Modify the learning rate according to a schedule.

        Args:
            base_lr: initial learning rate.
            epoch: the current epoch.
        Returns:
            The scheduled value for the learning rate.
        """


class ModuleProtocol(Protocol[_Input_contra, _Output_co]):
    """Protocol for a PyTorch module with type annotations."""

    def forward(self, inputs: _Input_contra) -> _Output_co:
        """Forward run of the network."""


class MetricCalculatorProtocol(Protocol[_Output_contra, _Target_contra]):
    """Protocol that calculates and returns metrics and loss."""

    @abc.abstractmethod
    def update(self,
               outputs: _Output_contra,
               targets: _Target_contra) -> Any:
        """
        Compute the metrics only.

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


class LossCalculatorProtocol(Protocol[_Output_contra, _Target_contra]):
    """Protocol that calculates and returns metrics and loss."""

    def forward(self,
                outputs: _Output_contra,
                targets: _Target_contra) -> torch.Tensor:
        """
        Process the outputs and targets and returns the loss.

        Args:
            outputs: model outputs.
            targets: ground truth.

        Returns:
            The computed loss.
        """

    @abc.abstractmethod
    def update(self,
               outputs: _Output_contra,
               targets: _Target_contra) -> Any:
        """
        Compute the metrics only.

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


class LearningProtocol(Protocol):
    """
    Protocol with specifications for the learning algorithm.

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
    clip_strategy: Callable[[Iterable[torch.nn.Parameter]], None]


@runtime_checkable
class ModelProtocol(Protocol[_Input_contra, _Output_co]):
    """
    Protocol for a wrapper around a torch module.

    Attributes:
        module: Pytorch module to optimize.
        epoch: the number of epochs the model has been trained so far.
        checkpoint: the object responsible for saving and loading the model.
    """
    module: torch.nn.Module
    epoch: int
    checkpoint: CheckpointProtocol

    @abc.abstractmethod
    def __call__(self, inputs: _Input_contra) -> _Output_co:
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

    def register_model(self, model: ModelProtocol):
        """Register the model to manage."""

    def register_optimizer(self, optimizer: torch.optim.Optimizer):
        """Register the optimizer connected to the model."""

    def save(self) -> None:
        """Save the model and optimizer state dictionaries."""

    def load(self, epoch: int = -1) -> None:
        """Load the model and optimizer state dictionaries."""


class EvaluationProtocol(Protocol[_Input, _Target, _Output]):
    """
    Protocol for a class that validates a model.

    Attributes:
        model: the model to evaluate.
        objective: object that calculates the metrics
    """
    model: ModelProtocol[_Input, _Output]
    objective: MetricCalculatorProtocol[_Output, _Target]

    @property
    def name(self) -> str:
        """The name of the model."""


@runtime_checkable
class TrainerProtocol(Protocol[_Input, _Target, _Output]):
    """
    Protocol for a class that train a model.

    Attributes:
        model: the model to train.
        objective: object that calculates the metrics and loss
    """
    model: ModelProtocol[_Input, _Output]
    learning_scheme: LearningProtocol
    objective: LossCalculatorProtocol[_Output, _Target]
    validation: EvaluationProtocol[_Input, _Target, _Output] | None

    @property
    def name(self) -> str:
        """The name of the model."""

    @property
    def terminated(self) -> bool:
        """If true, this trainer should not be used for training anymore."""

    def train(self, num_epochs: int) -> None:
        """
        Train the module for the specified number of epochs.

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
            base_lr: Optional[float],
            scheduler: Optional[SchedulerProtocol],
    ) -> None:
        """
        Update the learning rate(s).

        It updates the learning rates for each parameter's group in the
        optimizer based on input learning rate and scheduler.

        Args:
            base_lr: the initial learning rate.
            scheduler: scheduler for the learning rate.
        """

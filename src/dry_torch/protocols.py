"""This module defines internal protocols."""
from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping, MutableSequence
from typing import Any, Optional, Protocol, SupportsIndex, TypeAlias, TypeVar
from typing import runtime_checkable

import torch
from torch.utils import data

_T = TypeVar('_T')

Tensors: TypeAlias = torch.Tensor | MutableSequence[torch.Tensor]


@runtime_checkable
class NamedTupleProtocol(Protocol[_T]):
    """
    Optional protocol for the input and target types.

    Correctly handled by the default collate function.
    NamedTuples with different values are currently interpreted as Generic[Any].
    At the moment, this protocol won't support these interfaces
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
        """returns an iterator over the dataset in batches"""

    def __len__(self) -> int:
        """returns the number of batches in the dataset"""


class SchedulerProtocol(Protocol):
    """
    Protocol of a scheduler compatible with the LearningScheme class.
    """

    def __call__(self, base_lr: float, epoch: int) -> float:
        """
        Modifies the learning rate according to a schedule.

        Args:
            base_lr: initial learning rate.
            epoch: the current epoch.
        Returns:
            scheduled value for the learning rate.
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

        Returns:
            return value will not be used.
        """

    @abc.abstractmethod
    def compute(self) -> Mapping[str, torch.Tensor] | torch.Tensor | None:
        """Return a Mapping from the metric names to the calculated values."""

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
              the computed loss.
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

        Returns:
            return value will not be used.
        """

    @abc.abstractmethod
    def compute(self) -> Mapping[str, torch.Tensor] | torch.Tensor | None:
        """Return a Mapping from the metric names to the calculated values."""

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


@runtime_checkable
class ModelProtocol(Protocol[_Input_contra, _Output_co]):
    """
    Protocol for a wrapper around a torch module.

    Attributes:
        module: Pytorch module to optimize.
        epoch: the number of epochs the model has been trained so far.
    """
    module: torch.nn.Module
    epoch: int

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""

    @property
    def name(self) -> str:
        """The name of the model."""

    @abc.abstractmethod
    def __call__(self, inputs: _Input_contra) -> _Output_co:
        """Calls the module forward method."""

    @abc.abstractmethod
    def increment_epoch(self):
        """Increments the epoch by 1."""


class EvaluationProtocol(Protocol[_Input, _Target, _Output]):
    """
    Protocol for a class that validates a model.

    Attributes:
        model: the model to evaluate.
        calculator: object that calculates the metrics
    """
    model: ModelProtocol[_Input, _Output]
    calculator: MetricCalculatorProtocol[_Output, _Target]


@runtime_checkable
class TrainerProtocol(Protocol[_Input, _Target, _Output]):
    """
    Protocol for a class that train a model.

    Attributes:
        model: the model to train.
        calculator: object that calculates the metrics and loss
    """
    model: ModelProtocol[_Input, _Output]
    learning_scheme: LearningProtocol
    calculator: LossCalculatorProtocol[_Output, _Target]
    validation: EvaluationProtocol[_Input, _Target, _Output] | None

    @property
    def terminated(self) -> bool:
        """Training has terminated."""

    def train(self, num_epochs: int) -> None:
        """Trains the model."""

    def terminate_training(self, reason: str) -> None:
        """Terminate the training."""

    def save_checkpoint(self) -> None:
        """Save the model weights, the optimizer state and the logs."""

    def load_checkpoint(self, epoch: int = -1) -> None:
        """Load the model weights, the optimizer state and the logs."""

    def update_learning_rate(
            self,
            base_lr: Optional[float],
            scheduler: Optional[SchedulerProtocol],
    ) -> None:
        """
        It updates the learning rates for each parameters' group in the
        optimizer based on input learning rate and scheduler.

        Args:
            base_lr: the initial learning rate.
            scheduler: scheduler for the learning rate.
        """

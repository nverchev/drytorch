"""This module defines internal protocols."""

import abc
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Self
from typing import SupportsIndex, Type, TypeAlias, TypeVar, runtime_checkable

import torch
from torch.utils import data

from dry_torch.descriptors import Tensors

_T = TypeVar('_T')


@runtime_checkable
class NamedTupleProtocol(Protocol[_T]):
    """
    Optional protocol for the input and target types.

    Correctly handled by the default collate function.
    NamedTuples with different values are currently interpreted as Generic[Any].
    At the moment, this protocol won't support these interfaces
    """

    def __getitem__(self, index: SupportsIndex) -> _T:
        ...

    def __len__(self) -> int:
        ...

    def _asdict(self) -> dict[str, _T]:
        ...

    @property
    def _fields(self) -> tuple[str, ...]:
        ...

    @classmethod
    def _make(cls: Type[Self],
              *args: Any,
              **kwargs: Any) -> Self:
        """Create a new instance of the class."""


@runtime_checkable
class HasToDictProtocol(Protocol):
    """Optional protocol for the output type."""

    @abc.abstractmethod
    def to_dict(self) -> Mapping[str, torch.Tensor | Iterable[torch.Tensor]]:
        """Method to store outputs in the DryTorchDictList class"""


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
        ...


class TensorCallable(Protocol[_Output_contra, _Target_contra]):
    """Protocol for loss and metrics functions."""

    def __call__(self,
                 outputs: _Output_contra,
                 targets: _Target_contra) -> torch.Tensor:
        ...


class MetricsCalculatorProtocol(Protocol[_Output_contra, _Target_contra]):
    """Protocol that calculates and returns metrics."""

    @abc.abstractmethod
    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> None:
        """Compute the metrics."""

    @property
    @abc.abstractmethod
    def metrics(self) -> Mapping[str, torch.Tensor]:
        """Return a Mapping with the metric name and the calculated value."""

    @abc.abstractmethod
    def reset_calculated(self) -> None:
        """Delete the calculated values."""


class LossCalculatorProtocol(
    Protocol[_Output_contra, _Target_contra]
):
    """Protocol that calculates metrics and the final loss (criterion)."""

    @abc.abstractmethod
    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> None:
        """Compute the metrics and the final loss."""

    @property
    @abc.abstractmethod
    def metrics(self) -> Mapping[str, torch.Tensor]:
        """Return a Mapping with the metric name and the calculated value."""

    @property
    @abc.abstractmethod
    def criterion(self) -> torch.Tensor:
        """Return a tensor with the final loss value."""

    @abc.abstractmethod
    def reset_calculated(self) -> None:
        """Delete the calculated values."""


@runtime_checkable
class ModelProtocol(Protocol[_Input_contra, _Output_co]):
    """
    Protocol for a wrapper around a torch module.

    Attributes:
        name: name of the model.
        module: underlying PyTorch module.
    """
    name: str
    module: torch.nn.Module

    @property
    def device(self) -> torch.device:
        """Returns the device of the module."""

    @abc.abstractmethod
    def __call__(self, inputs: _Input_contra) -> _Output_co:
        """Calls the module forward method."""


@runtime_checkable
class TrainerProtocol(Protocol):
    """
    Protocol for a class that train and validates a model.

    Attributes:
        model: the model to train.
    """
    model: ModelProtocol

    def validate(self) -> None:
        """Validates the model."""

    def terminate_training(self) -> None:
        """Terminate the training."""

    def save_checkpoint(self, replace_previous: bool = False) -> None:
        """Save the model weights, the optimizer state and the logs."""

    def load_checkpoint(self, epoch: int = -1) -> None:
        """Load the model weights, the optimizer state and the logs."""

    def update_learning_rate(self, learning_rate: float) -> None:
        """Update the learning rate."""


"""
This module defines internal protocols.

Protocols:
    NamedTupleProtocol: optional protocol for input and target types.
    HasToDictProtocol: optional protocol for the output type.
    LoaderProtocol: loads and batches a dataset.
    SchedulerProtocol: scheduler compatible with the LearningScheme class.
    ModuleProtocol: for a PyTorch module with type annotations.
    TensorProtocol: for loss and metrics functions.
    MetricProtocol: calculates and returns metrics.
    LossProtocol: calculates and returns metrics and the final loss tensor.
    ModuleProtocol: for a wrapper around a torch module.

"""

import abc
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Self
from typing import SupportsIndex, Type, TypeAlias, TypeVar, runtime_checkable

import torch
from dry_torch.descriptors import Tensors
from torch.utils import data

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
        ...


class HasToDictProtocol(Protocol):
    """
    Optional protocol for the output type.

    The to_dict method ensures that the outputs is correctly stored in the
    DryTorchDictList class.
    """

    @abc.abstractmethod
    def to_dict(self) -> Mapping[str, torch.Tensor | Iterable[torch.Tensor]]:
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

    Methods:
        __iter__: returns an iterator over the dataset in batches.
        __len__: returns the number of batches in the dataset.
    """
    batch_size: Optional[int]
    dataset: data.Dataset

    def __iter__(self) -> Iterator[_Data_co]:
        ...

    def __len__(self) -> int:
        ...


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
        ...


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
        ...

    @property
    @abc.abstractmethod
    def metrics(self) -> Mapping[str, torch.Tensor]:
        ...

    @abc.abstractmethod
    def reset_calculated(self) -> None:
        ...


class LossCalculatorProtocol(
    Protocol[_Output_contra, _Target_contra]
):
    """Protocol that calculates metrics and the final loss (criterion)."""

    @abc.abstractmethod
    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> None:
        ...

    @property
    @abc.abstractmethod
    def metrics(self) -> Mapping[str, torch.Tensor]:
        ...

    @property
    @abc.abstractmethod
    def criterion(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def reset_calculated(self) -> None:
        ...


@runtime_checkable
class ModelProtocol(Protocol[_Input_contra, _Output_co]):
    """
    Protocol for a wrapper around a torch module.

    Attributes:
        name: name of the model
        module: underlying PyTorch module

    """
    name: str
    module: torch.nn.Module

    @property
    def device(self) -> torch.device:
        ...

    @abc.abstractmethod
    def __call__(self, inputs: _Input_contra) -> _Output_co:
        ...

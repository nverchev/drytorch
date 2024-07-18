from __future__ import annotations

import pathlib
import abc
from typing import Protocol, TypedDict, Iterator, Callable, TypeVar, TypeAlias, \
    Optional
from typing import Any, Union, Type, Self, runtime_checkable, SupportsIndex
from typing import Iterable, Mapping
import enum

import pandas as pd
import torch
from torch.nn.parameter import Parameter
from torch.utils import data


class Split(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


class OptParams(TypedDict):
    params: Iterator[Parameter]
    lr: float


class StatePath(TypedDict):
    state: pathlib.Path
    optimizer: pathlib.Path


PartitionsLength: TypeAlias = dict[Split, int]
LogsDict: TypeAlias = dict[Split, pd.DataFrame]
PathDict: TypeAlias = dict[Split, pathlib.Path]

Tensors: TypeAlias = Union[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    list[torch.Tensor],
]

_T = TypeVar('_T')

"""
Correctly handled by the default collate function.
NamedTuples with different values are currently interpreted as Generic of Any.
At the moment, this protocol won't support these interfaces
"""


@runtime_checkable
class NamedTupleProtocol(Protocol[_T]):

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
    def _make(cls: Type[NamedTupleProtocol],
              *args: Any,
              **kwargs: Any) -> NamedTupleProtocol:
        ...


class HasToDictProtocol(Protocol):
    @abc.abstractmethod
    def to_dict(self) -> Mapping[str, torch.Tensor | Iterable[torch.Tensor]]:
        ...


InputType: TypeAlias = Tensors | NamedTupleProtocol[Tensors]
OutputType: TypeAlias = Any
TargetType: TypeAlias = Tensors | NamedTupleProtocol[Tensors]

_Input_co = TypeVar('_Input_co', bound=InputType, covariant=True)
_Target_co = TypeVar('_Target_co', bound=TargetType, covariant=True)

_Output_co = TypeVar('_Output_co', bound=OutputType, covariant=True)

_Data_co = TypeVar('_Data_co',
                   bound=tuple[InputType, TargetType],
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
    batch_size: Optional[int]
    dataset: data.Dataset

    def __iter__(self) -> Iterator[_Data_co]:
        ...

    def __len__(self) -> int:
        ...


class SchedulerProtocol(Protocol):
    """
    Protocol of a scheduler compatible with the Model class.
    """

    def __call__(self, base_lr: float, epoch: int) -> float:
        ...


class ModuleProtocol(Protocol[_Input_contra, _Output_co]):

    def forward(self, inputs: _Input_contra) -> _Output_co:
        ...


class TensorCallable(Protocol[_Output_contra, _Target_contra]):

    def __call__(self,
                 outputs: _Output_contra,
                 targets: _Target_contra) -> torch.Tensor:
        ...


class MetricsCalculatorProtocol(Protocol[_Output_contra, _Target_contra]):
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


class LearningProtocol(Protocol):
    """
        optimizer_cls: the optimizer class to bind_to_model to the module.
         Defaults to torch.optim.Adam.
        lr: a dictionary of learning rates for the named parameters or a float
        for a global value.
        optimizer_defaults: optional arguments for the optimizer
        (same for all the parameters).
        scheduler: modifies the learning rate given the current epoch. Default
        value does not implement a scheduler.
    """
    optimizer_cls: Type[torch.optim.Optimizer]
    lr: float | dict[str, float]
    optimizer_defaults: dict[str, Any]
    scheduler: SchedulerProtocol


@runtime_checkable
class ModelProtocol(Protocol[_Input_contra, _Output_co]):
    name: str
    device: torch.device
    module: torch.nn.Module

    @abc.abstractmethod
    def __call__(self, inputs: _Input_contra) -> _Output_co:
        ...


class TrainerProtocol(Protocol):

    def train(self, num_epochs: int) -> None:
        ...

    def terminate_training(self) -> None:
        ...

    def add_pre_epoch_hook(self, hook: Callable[[Self], None]) -> None:
        ...

    def add_post_epoch_hook(self, hook: Callable[[Self], None]) -> None:
        ...

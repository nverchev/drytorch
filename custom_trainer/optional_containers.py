from __future__ import annotations

from typing import Any, Iterator

import torch
from torch import Tensor

from .exceptions import NotATensorError


class PreferablySlotted:
    """
    A base TypedDict-like class that leverages slots.
    This class provides a mechanism to define the slots for its subclasses,
    in case the user did not define them.
    It also defines a reset value for the slots and a clear method.
    """
    __slots__: tuple[str, ...] = tuple()
    reset_value: Any = None

    @classmethod
    def __new__(cls, *args, **kwargs) -> PreferablySlotted:
        instance = super().__new__(cls)
        null = object()
        # Compatible with non-slotted classes.
        if getattr(instance, '__dict__', null) != null:
            instance.__slots__ = tuple(kwargs.keys())
        return instance

    def __init__(self, /, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def clear(self) -> None:
        for name in self.__slots__:
            setattr(self, name, self.reset_value)


class BaseMetricsContainer(PreferablySlotted):
    """
    A base class to store batched values of the metrics as torch Tensors. Needs to be subclassed.
    To improve performance, define __slots__ with the metrics' names.
    """
    __slots__: tuple[str, ...] = tuple()
    reset_value = torch.tensor(0.0)

    def __init__(self, **kwargs: torch.Tensor) -> None:
        super().__init__(**kwargs)

    @property
    def metrics(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in self.__slots__ if name != 'criterion'}


class BaseLossAndMetricsContainer(BaseMetricsContainer):
    """
    A base class to store the batched values of the loss and other metrics as torch Tensors. Needs to be subclassed.
    To improve performance, define __slots__ with the 'criterion' and the other metrics' names.
    """
    __slots__: tuple[str, ...] = ('criterion',)
    criterion: torch.Tensor

    def __init__(self, criterion: torch.FloatTensor, **kwargs: torch.Tensor) -> None:
        self.criterion: torch.FloatTensor = criterion
        super().__init__(**kwargs)


class OutputsContainer(PreferablySlotted):
    """
    A base TypedDict-like base class to store the outputs values. Needs to be subclassed.
    To improve performance, define __slots__ with the 'outputs' names.
    """
    __slots__: tuple[str, ...] = tuple()
    reset_value = torch.tensor(0.0)

    def __init__(self, **kwargs: torch.Tensor | list[torch.Tensor]) -> None:
        super().__init__(**kwargs)

    def __iter__(self) -> Iterator[tuple[str, Tensor | list[Tensor]]]:
        for name in self.__slots__:
            yield name, getattr(self, name)

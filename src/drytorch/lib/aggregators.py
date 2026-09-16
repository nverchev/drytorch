"""Accumulators and a container merging them by name."""

from __future__ import annotations

import abc
import copy
import dataclasses

from collections.abc import Mapping
from typing import Any, Final, Generic, Self, TypeVar

import torch

from torch import distributed as dist
from typing_extensions import override


__all__ = [
    'AbstractAccumulator',
    'Aggregator',
    'TorchMeanAccumulator',
]


_T_contra = TypeVar('_T_contra', contravariant=True)
_R_co = TypeVar('_R_co', covariant=True)


class AbstractAccumulator(abc.ABC, Generic[_T_contra, _R_co]):
    """Stateful aggregation container."""

    @abc.abstractmethod
    def merge(self, other: Any) -> None:
        """Merge another accumulator into this one.

        Args:
            other: an accumulator of the same class (annotated accordingly).
        """

    @abc.abstractmethod
    def reduce(self) -> _R_co:
        """Return reduced value."""

    @abc.abstractmethod
    def sync(self) -> None:
        """Synchronize state across distributed processes."""

    @classmethod
    @abc.abstractmethod
    def from_value(cls, value: _T_contra) -> Self:
        """Create an accumulator from raw value."""


_A = TypeVar('_A', bound=AbstractAccumulator[Any, Any])


class Aggregator(Generic[_A]):
    """Container merging named accumulators of the same type.

    Attributes:
        accumulators: the accumulators by name.
    """

    __slots__ = ('_accumulators',)

    _accumulators: Final[dict[str, _A]]

    def __init__(self, **accumulators: _A) -> None:
        """Initialize.

        Args:
            accumulators: named accumulators to store.
        """
        self._accumulators = dict(accumulators)
        return

    @property
    def accumulators(self) -> Mapping[str, _A]:
        """Return the accumulators by name."""
        return self._accumulators

    def __bool__(self) -> bool:
        """Return True if any accumulators are stored."""
        return bool(self.accumulators)

    def __repr__(self) -> str:
        """Return representation of the aggregator."""
        return (
            f'{self.__class__.__name__}(keys={list(self.accumulators.keys())})'
        )

    def clear(self) -> None:
        """Remove all accumulators."""
        self._accumulators.clear()
        return

    def keys(self) -> list[str]:
        """Return stored metric names."""
        return list(self.accumulators.keys())

    def merge(self, other: Mapping[str, _A]) -> None:
        """Merge the named accumulators into this aggregator.

        Accumulators with a name not yet stored are copied.

        Args:
            other: the accumulators to merge by name.
        """
        for key, acc in other.items():
            if key in self._accumulators:
                self._accumulators[key].merge(acc)
            else:
                self._accumulators[key] = copy.deepcopy(acc)

        return

    def sync(self) -> None:
        """Synchronize the accumulators across distributed processes."""
        for acc in self.accumulators.values():
            acc.sync()

        return


@dataclasses.dataclass(slots=True)
class TorchMeanAccumulator(AbstractAccumulator[torch.Tensor, torch.Tensor]):
    """Accumulator computing arithmetic mean for tensors.

    Attributes:
        count: number of values.
        total: sum of all values.
    """

    count: int
    total: torch.Tensor

    @override
    def merge(self, other: TorchMeanAccumulator) -> None:
        self.total += other.total
        self.count += other.count
        return

    @override
    def reduce(self) -> torch.Tensor:
        return self.total / self.count

    @override
    def sync(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
            count_tensor = torch.tensor(
                self.count,
                device=self.total.device,
                dtype=torch.long,
            )
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            self.count = int(count_tensor.item())

        return

    @classmethod
    @override
    def from_value(cls, value: torch.Tensor) -> TorchMeanAccumulator:
        return cls(
            count=value.numel(),
            total=value.detach().sum(),
        )

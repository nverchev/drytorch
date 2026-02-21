"""Module containing classes for aggregating values."""

from __future__ import annotations

import abc
import collections
import copy

from collections.abc import KeysView, Mapping
from typing import Any, Final, Generic, Self, TypeVar

import torch

from torch import distributed as dist
from typing_extensions import override


_T = TypeVar('_T', torch.Tensor, float)


class AbstractAverager(Generic[_T], metaclass=abc.ABCMeta):
    """Average tensor values from dict-like objects.

    It registers sample size to calculate the precise sample average.

    Attributes:
        aggregate: a dictionary with the aggregated values.
        counts: a dictionary with the count of the total elements.
    """

    __slots__: Final = ('_cached_reduce', 'aggregate', 'counts')

    aggregate: dict[str, _T]
    counts: collections.defaultdict[str, int]
    _cached_reduce: dict[str, _T]

    def __init__(self, **kwargs: _T):
        """Initialize.

        Args:
            kwargs: named values to average.
        """
        self.aggregate = {}
        self.counts = collections.defaultdict(int)
        self.__iadd__(kwargs)
        self._cached_reduce = {}

    def __add__(self, other: AbstractAverager[_T] | Mapping[str, _T]) -> Self:
        """Join current data with data from another Averager.

        Args:
            other: the other Averager.
        """
        if isinstance(other, Mapping):
            other = self.__class__(**other)

        out = copy.deepcopy(self)
        out += other
        return out

    def __bool__(self) -> bool:
        """Return True if data is present."""
        return bool(self.aggregate)

    def __deepcopy__(self, memo: dict[int, Any] | None) -> Self:
        """Deep copy magic method.

        Args:
            memo: Dictionary of already copied objects.

        Returns:
            A deep copy of the object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.aggregate = copy.deepcopy(self.aggregate)
        result.counts = copy.copy(self.counts)
        result._cached_reduce = {}
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.aggregate == other.aggregate and self.counts == other.counts

    def __iadd__(self, other: AbstractAverager[_T] | Mapping[str, _T]) -> Self:
        """Merge current data with data from another Averager.

        Args:
            other: the other Averager.
        """
        if isinstance(other, AbstractAverager):
            other_aggregate = other.aggregate
            other_counts = other.counts
        else:
            other_aggregate = {}
            other_counts = collections.defaultdict[str, int]()
            for key, value in other.items():
                other_aggregate[key] = self._aggregate(value)
                other_counts[key] = self._count(value)

        if self.aggregate:  # fail if new elements are added after the start
            for key, value in other_aggregate.items():
                self.aggregate[key] += value
                self.counts[key] += other_counts[key]
        else:
            self.aggregate.update(other_aggregate)
            self.counts.update(other_counts)

        self._cached_reduce = {}
        return self

    @override
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(counts={self.counts})'

    def clear(self) -> None:
        """Clear data contained in the class."""
        self._cached_reduce.clear()
        self.aggregate.clear()
        self.counts.clear()
        return

    def keys(self) -> KeysView[str]:
        """Calculate the count of a value."""
        return self.aggregate.keys()

    def reduce(self) -> dict[str, _T]:
        """Return the averaged values."""
        if not self._cached_reduce:
            self._cached_reduce = {
                key: self._reduce(value, self.counts[key])
                for key, value in self.aggregate.items()
            }

        return self._cached_reduce

    def all_reduce(self) -> dict[str, _T]:
        """Synchronize the averaged values across processes."""
        self._cached_reduce.clear()  # make sure to recalculate all the values
        return self.reduce()

    @staticmethod
    def _reduce(aggregated: _T, count: int) -> _T:
        return aggregated / count

    @staticmethod
    @abc.abstractmethod
    def _aggregate(value: _T) -> _T: ...

    @staticmethod
    @abc.abstractmethod
    def _count(value: _T) -> int: ...


class Averager(AbstractAverager[float]):
    """Subclass of Aggregator with an implementation for float values."""

    @staticmethod
    @override
    def _aggregate(value: float) -> float:
        return value

    @staticmethod
    @override
    def _count(value: float) -> int:
        return 1


class TorchAverager(AbstractAverager[torch.Tensor]):
    """Subclass of Aggregator with an implementation for torch.Tensor."""

    @staticmethod
    @override
    def _aggregate(value: torch.Tensor) -> torch.Tensor:
        return value.detach().sum()

    @staticmethod
    @override
    def _count(value: torch.Tensor) -> int:
        return value.numel()

    @override
    def all_reduce(self) -> dict[str, torch.Tensor]:
        """Synchronize the values across processes."""
        if dist.is_available() and torch.distributed.is_initialized():
            for key, value in self.aggregate.items():
                torch.distributed.all_reduce(
                    value, op=torch.distributed.ReduceOp.SUM
                )
                count_tensor = torch.tensor(
                    self.counts[key], device=value.device, dtype=torch.long
                )
                torch.distributed.all_reduce(
                    count_tensor, op=torch.distributed.ReduceOp.SUM
                )
                self.counts[key] = int(count_tensor.item())

        return super().all_reduce()

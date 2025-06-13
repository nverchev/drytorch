"""Module containing classes to aggregate and average samples."""

from __future__ import annotations

import abc
import collections
from collections.abc import KeysView, Mapping
import copy
from typing import Generic, Self, TypeVar
from typing_extensions import override

import torch

_T = TypeVar('_T', torch.Tensor, float)


class Aggregator(Generic[_T], metaclass=abc.ABCMeta):
    """
    Average tensor values from dict-like objects.

    It registers sample size to calculate precise sample average.

    Attributes:
        aggregate: a dictionary with the aggregated values.
        counts: a dictionary with the count of the total elements.
    """
    __slots__ = ('aggregate', 'counts', '_cached_reduce')

    def __init__(self, **kwargs: _T):
        """
        Args:
            kwargs: named values to average.
        """
        self.aggregate: dict[str, _T] = {}
        self.counts = collections.defaultdict[str, int](int)
        self.__iadd__(kwargs)
        self._cached_reduce: dict[str, _T] = {}

    def __add__(self, other: Aggregator | Mapping[str, _T]) -> Self:
        if isinstance(other, Mapping):
            other = self.__class__(**other)

        out = copy.deepcopy(self)
        out += other
        return out

    def __bool__(self) -> bool:
        return bool(self.aggregate)

    def __deepcopy__(self, memo: dict) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.aggregate = copy.deepcopy(self.aggregate)
        result.counts = copy.copy(self.counts)
        result._cached_reduce = {}
        for k, v in self.__dict__.items():
            result.__dict__[k] = copy.deepcopy(v, memo)

        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.aggregate == other.aggregate and self.counts == other.counts

    def __iadd__(self, other: Aggregator | Mapping[str, _T]) -> Self:
        if isinstance(other, Aggregator):
            other_aggregate = other.aggregate
            other_counts = other.counts
        else:
            other_aggregate = {}
            other_counts = collections.defaultdict[str, int]()
            for key, value in other.items():
                other_aggregate[key] = self._aggregate(value)
                other_counts[key] = self._count(value)

        if self.aggregate:  # fail if new elements are added after start
            for key, value in other_aggregate.items():
                self.aggregate[key] += value
                self.counts[key] += other_counts[key]
        else:
            self.aggregate = other_aggregate
            self.counts = other_counts

        self._cached_reduce = {}
        return self

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
            self._cached_reduce = {key: self._reduce(value, self.counts[key])
                                   for key, value in self.aggregate.items()}

        return self._cached_reduce

    @staticmethod
    @abc.abstractmethod
    def _aggregate(value: _T) -> _T:
        ...

    @staticmethod
    @abc.abstractmethod
    def _count(value: _T) -> int:
        ...

    @staticmethod
    def _reduce(aggregated: _T, count: int) -> _T:
        return aggregated / count


class Averager(Aggregator[float]):
    """Subclass of Aggregator with an implementation for float values."""

    @staticmethod
    @override
    def _aggregate(value: float) -> float:
        return value

    @staticmethod
    @override
    def _count(value: float) -> int:
        return 1


class TorchAverager(Aggregator[torch.Tensor]):
    """Subclass of Aggregator with an implementation for torch.Tensor."""

    @staticmethod
    @override
    def _aggregate(value: torch.Tensor) -> torch.Tensor:
        return value.sum()

    @staticmethod
    @override
    def _count(value: torch.Tensor) -> int:
        return value.numel()

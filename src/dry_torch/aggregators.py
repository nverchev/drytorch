"""This module contains the Aggregator abstract class and TorchAggregator."""

import abc
import collections
from collections.abc import Mapping, Iterable, KeysView
import copy
from typing import Self, TypeVar, Generic
from typing_extensions import override

import torch

from src.dry_torch import exceptions

_T = TypeVar('_T', torch.Tensor, float)


class Aggregator(Generic[_T], metaclass=abc.ABCMeta):
    """
    This class averages tensor values from dict-like objects.

    It keeps count of the number of samples in the arguments and allows a
    precise sample average in case of unequal sample sizes.

    Args:
        iterable: a _T valued dictionary or equivalent iterable.
        kwargs: keyword _T arguments (alternative syntax).

    Attributes:
        aggregate: a dictionary with the aggregated values.
        counts: a dictionary with the count of the total elements.
    """
    __slots__ = ('aggregate', 'counts', '_cached_reduce')

    def __init__(self,
                 iterable: Iterable[tuple[str, _T]] = (),
                 /,
                 **kwargs: _T):
        self.aggregate: dict[str, _T] = {}
        self.counts = collections.defaultdict[str, int](int)
        for key, value in iterable:
            self[key] = value

        for key, value in kwargs.items():
            self[key] = value

        self._cached_reduce: dict[str, _T] = {}

    def __add__(self, other: Self | Mapping[str, _T]) -> Self:
        if isinstance(other, Mapping):
            other = self.__class__(**other)
        out = copy.deepcopy(self)
        out += other
        return out

    def __iadd__(self, other: Self | Mapping[str, _T]) -> Self:
        if isinstance(other, Mapping):
            other = self.__class__(**other)
        if self.aggregate:  # fail if new elements are added after start
            for key, value in other.aggregate.items():
                self.aggregate[key] += value
        else:
            self.aggregate = other.aggregate

        for key, count in other.counts.items():
            self.counts[key] += count
        return self

    def __setitem__(self, key: str, value: _T) -> None:
        self.aggregate[key] = self._aggregate(value)
        self.counts[key] = self._count(value)
        return

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.aggregate == other.aggregate and self.counts == other.counts

    def clear(self) -> None:
        """Clear data contained in the class."""
        self._cached_reduce = {}
        self.aggregate.clear()
        self.counts.clear()
        return

    def keys(self) -> KeysView[str]:
        """Calculate the count of a value."""
        return self.aggregate.keys()

    def reduce(self) -> dict[str, _T]:
        """Return the averages values."""
        if not self._cached_reduce:
            self._cached_reduce = {key: self._reduce(value, self.counts[key])
                                   for key, value in self.aggregate.items()}
        return self._cached_reduce

    def __bool__(self) -> bool:
        return bool(self.aggregate)

    @staticmethod
    def _reduce(aggregated: _T, count: int) -> _T:
        return aggregated / count

    @staticmethod
    @abc.abstractmethod
    def _count(value: _T) -> int:
        ...

    @staticmethod
    @abc.abstractmethod
    def _aggregate(value: _T) -> _T:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(counts={self.counts})'


class Averager(Aggregator[float]):
    """ Subclass of Aggregator with an implementation for torch.Tensor.

    It accepts only s with no more than one non-squeezable dimension,
     typically the one for the batch. """

    @staticmethod
    @override
    def _count(value: float) -> int:
        return 1

    @staticmethod
    @override
    def _aggregate(value: float) -> float:
        return value


class TorchAverager(Aggregator[torch.Tensor]):
    """ Subclass of Aggregator with an implementation for torch.Tensor.

    It accepts only s with no more than one non-squeezable dimension,
     typically the one for the batch. """

    @staticmethod
    @override
    def _count(value: torch.Tensor) -> int:
        return value.numel()

    @staticmethod
    @override
    def _aggregate(value: torch.Tensor) -> torch.Tensor:
        if value.dim() > 1:
            raise exceptions.MetricsNotAVectorError(list(value.shape))
        else:
            return value.sum(0)

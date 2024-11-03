"""This module contains the Aggregator abstract class and TorchAggregator."""

import abc
import collections
from collections.abc import Mapping, Iterable, KeysView
from typing import Self, TypeVar, Generic
from typing_extensions import override

import torch

from src.dry_torch import exceptions

_T = TypeVar('_T')


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
    __slots__ = ('aggregate', 'counts')

    def __init__(self,
                 iterable: Iterable[tuple[str, _T]] = (),
                 /,
                 **kwargs: _T):
        self.aggregate = collections.defaultdict[str, float](float)
        self.counts = collections.defaultdict[str, int](int)
        for key, value in iterable:
            self[key] = value

        for key, value in kwargs.items():
            self[key] = value

    def __add__(self, other: Self | Mapping[str, _T]) -> Self:
        if isinstance(other, Mapping):
            other = self.__class__(**other)
        out = self.__copy__()
        out += other
        return out

    def __iadd__(self, other: Self | Mapping[str, _T]) -> Self:
        if isinstance(other, Mapping):
            other = self.__class__(**other)

        for key, value in other.aggregate.items():
            self.aggregate[key] += value

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

    @property
    def first_metric(self):
        try:
            return next(iter(self.keys()))
        except StopIteration as si:
            raise exceptions.MetricNotFoundError(self.__class__.__name__,
                                                 'yet') from si

    def clear(self) -> None:
        """Clear data contained in the class."""
        self.aggregate.clear()
        self.counts.clear()
        return

    def keys(self) -> KeysView[str]:
        """Calculate the count of a value."""
        return self.aggregate.keys()

    def reduce(self, key: str) -> float:
        return self.aggregate[key] / self.counts[key]

    def reduce_all(self) -> dict[str, float]:
        """Return the averages values as floating points."""
        return {key: value / self.counts[key]
                for key, value in self.aggregate.items()}

    def __copy__(self) -> Self:
        copied = self.__class__()
        copied.aggregate = self.aggregate.copy()
        copied.counts = self.counts.copy()
        return copied

    def __bool__(self) -> bool:
        return bool(self.aggregate)

    @staticmethod
    @abc.abstractmethod
    def _count(value: _T) -> int:
        ...

    @staticmethod
    @abc.abstractmethod
    def _aggregate(value: _T) -> float:
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
    def _aggregate(value: torch.Tensor) -> float:
        try:
            return value.sum(0).item()
        except RuntimeError:
            raise exceptions.MetricsNotAVectorError(list(value.shape))

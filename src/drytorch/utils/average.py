"""Module containing classes and functions to average samples."""

from __future__ import annotations

import abc
import collections
import copy
import math

from collections.abc import Callable, KeysView, Mapping, Sequence
from typing import Any, Final, Generic, Self, TypeVar

import torch

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

    def __init__(self, **kwargs: _T):
        """Constructor.

        Args:
            kwargs: named values to average.
        """
        self.aggregate: dict[str, _T] = {}
        self.counts: collections.defaultdict[str, int] = (
            collections.defaultdict(int)
        )
        self.__iadd__(kwargs)
        self._cached_reduce: dict[str, _T] = {}

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
            result.__dict__[k] = copy.deepcopy(v, memo)

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

    @staticmethod
    @abc.abstractmethod
    def _aggregate(value: _T) -> _T: ...

    @staticmethod
    @abc.abstractmethod
    def _count(value: _T) -> int: ...

    @staticmethod
    def _reduce(aggregated: _T, count: int) -> _T:
        return aggregated / count


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
        return value.sum()

    @staticmethod
    @override
    def _count(value: torch.Tensor) -> int:
        return value.numel()


def get_moving_average(
    decay: float = 0.9,
    mass_coverage: float = 0.99,
) -> Callable[[Sequence[float]], float]:
    """Return a moving average by specifying the decay.

    Args:
        decay: the closer to 0, the more the last elements have weight.
        mass_coverage: cumulative weight proportion before tail dropping.

    Returns:
        The moving average function.

    Raises:
        ValueError if the decay is not between 0 and 1.
        ValueError if the mass_coverage is not between 1 - decay and 1.
    """
    if not 0 < decay < 1:
        raise ValueError('decay must be between 0 and 1.')

    if not 1 - decay <= mass_coverage <= 1:
        raise ValueError('mass_coverage should be between 1 - decay and 1.')

    # how far back to go back before the weight drops below the threshold
    if mass_coverage < 1:
        stop = -int(math.log(1 - mass_coverage, decay)) - 2
    else:
        stop = None

    def _mean(float_list: Sequence[float], /) -> float:
        total: float = 0
        total_weights: float = 0  # should get close to one
        weight = 1 - decay  # weights are normalized
        for elem in float_list[:stop:-1]:
            total += weight * elem
            total_weights += weight
            weight *= decay

        return total / total_weights

    repr_mean = f'moving_average(decay={decay}, mass_coverage={mass_coverage})'
    _mean.__name__ = repr_mean
    return _mean


def get_trailing_mean(window_size: int) -> Callable[[Sequence[float]], float]:
    """Return a trailing average by specifying window size.

    Args:
        window_size: number of items to aggregate.

    Returns:
        The windowed average function.

    Raises:
        ValueError if the window size is negative.
    """
    if window_size <= 0:
        raise ValueError('window_size must be positive.')

    def _mean(float_list: Sequence[float], /) -> float:
        clipped_window = min(window_size, len(float_list))
        return sum(float_list[-clipped_window:]) / clipped_window

    repr_mean = f'trailing_mean(window_size={window_size})'
    _mean.__name__ = repr_mean
    return _mean

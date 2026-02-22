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


class AbstractAggregator(Generic[_T], metaclass=abc.ABCMeta):
    """Aggregate values from dict-like objects.

    Attributes:
        partials: dictionary of aggregated values.
        counts: dictionary of counts for each aggregated quantity.
    """

    __slots__: Final = ('_cached_reduce', 'counts', 'partials')

    partials: dict[str, _T]
    counts: collections.defaultdict[str, int]
    _cached_reduce: dict[str, _T]

    def __init__(self, **kwargs: _T) -> None:
        """Initialize.

        Args:
            kwargs: named values to average.
        """
        self.partials = {}
        self.counts = collections.defaultdict(int)
        self.__iadd__(kwargs)
        self._cached_reduce = {}

    def __add__(self, other: AbstractAggregator[_T] | Mapping[str, _T]) -> Self:
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
        return bool(self.partials)

    def __deepcopy__(self, memo: dict[int, Any] | None) -> Self:
        """Deep copy magic method.

        Args:
            memo: Dictionary of copied objects.

        Returns:
            A deep copy of the object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.partials = copy.deepcopy(self.partials)
        result.counts = copy.copy(self.counts)
        result._cached_reduce = {}
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return self.partials == other.partials and self.counts == other.counts

    def __iadd__(
        self, other: AbstractAggregator[_T] | Mapping[str, _T]
    ) -> Self:
        """Merge current data with data from another Averager.

        Args:
            other: the other Averager.
        """
        if isinstance(other, AbstractAggregator):
            other_sums = other.partials
            other_counts = other.counts
        else:
            other_sums = {}
            other_counts = collections.defaultdict[str, int]()
            for key, value in other.items():
                other_sums[key] = self._aggregate(value)
                other_counts[key] = self._count(value)

        if self.partials:
            for key, value in other_sums.items():
                self.partials[key] += value
                self.counts[key] += other_counts[key]
        else:
            self.partials.update(other_sums)
            self.counts.update(other_counts)

        self._cached_reduce = {}
        return self

    @override
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(counts={self.counts})'

    def clear(self) -> None:
        """Clear all accumulated data."""
        self._cached_reduce.clear()
        self.partials.clear()
        self.counts.clear()

    def keys(self) -> KeysView[str]:
        """Names of the aggregated quantities."""
        return self.partials.keys()

    def reduce(self) -> dict[str, _T]:
        """Return the reduced values."""
        if not self._cached_reduce:
            self._cached_reduce = {
                key: self._reduce(value, self.counts[key])
                for key, value in self.partials.items()
            }
        return self._cached_reduce

    def all_reduce(self) -> dict[str, _T]:
        """Synchronize accumulators across processes, then reduce."""
        self._cached_reduce.clear()
        return self.reduce()

    @staticmethod
    @abc.abstractmethod
    def _aggregate(value: _T) -> _T:
        """Convert a raw input value into its initial accumulator form."""

    @staticmethod
    @abc.abstractmethod
    def _count(value: _T) -> int:
        """Extracts the number of elements represented in value."""

    @staticmethod
    @abc.abstractmethod
    def _reduce(accumulated: _T, count: int) -> _T:
        """Produce the final result from the accumulator and element count."""


class AbstractAverager(AbstractAggregator[_T], abc.ABC):
    """Use mean as the reduction function."""

    @staticmethod
    @override
    def _reduce(accumulated: _T, count: int) -> _T:
        return accumulated / count


class Averager(AbstractAverager[float]):
    """Averager for plain Python floats."""

    @staticmethod
    @override
    def _aggregate(value: float) -> float:
        return value

    @staticmethod
    @override
    def _count(value: float) -> int:
        return 1


class TorchAverager(AbstractAverager[torch.Tensor]):
    """Averager for ``torch.Tensor`` with optional distributed support."""

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
        if dist.is_available() and dist.is_initialized():
            for key, value in self.partials.items():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                count_tensor = torch.tensor(
                    self.counts[key], device=value.device, dtype=torch.long
                )
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                self.counts[key] = int(count_tensor.item())

        return super().all_reduce()

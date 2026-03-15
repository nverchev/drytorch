"""Module containing generic accumulator-based aggregators."""

from __future__ import annotations

import abc
import copy

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Generic, Self, TypeVar

import torch

from torch import distributed as dist
from typing_extensions import override


_T = TypeVar('_T')
_R = TypeVar('_R')


class AbstractAccumulator(Generic[_T, _R], abc.ABC):
    """Stateful aggregation container."""

    @classmethod
    @abc.abstractmethod
    def from_value(cls, value: _T) -> AbstractAccumulator[_T, _R]:
        """Create accumulator from raw value."""

    @abc.abstractmethod
    def merge(self, other: AbstractAccumulator[_T, _R]) -> None:
        """Merge another accumulator into this one."""

    @abc.abstractmethod
    def reduce(self) -> _R:
        """Return reduced value."""

    @abc.abstractmethod
    def sync(self) -> None:
        """Synchronize state across distributed processes."""


class AbstractAggregator(Generic[_T, _R], metaclass=abc.ABCMeta):
    """Aggregate named values using accumulator objects."""

    __slots__: Final = ('_cached_reduce', 'accumulators')

    accumulator_cls: type[AbstractAccumulator[_T, _R]]
    accumulators: dict[str, AbstractAccumulator[_T, _R]]
    _cached_reduce: dict[str, _R]

    def __init__(self, **kwargs: _T) -> None:
        """Initialize.

        Args:
            kwargs: named values to aggregate.
        """
        self.accumulators = {}
        self._cached_reduce = {}
        for key, value in kwargs.items():
            self.accumulators[key] = self.accumulator_cls.from_value(value)

        return

    def __add__(self, other: Self | Mapping[str, _T]) -> Self:
        """Return new aggregator containing merged data."""
        result = copy.deepcopy(self)
        result += other
        return result

    def __bool__(self) -> bool:
        """Return True if any values are stored."""
        return bool(self.accumulators)

    def __iadd__(self, other: Self | Mapping[str, _T]) -> Self:
        """Merge another aggregator or mapping into this one."""
        if isinstance(other, Mapping):
            other = self.__class__(**other)

        for key, acc in other.accumulators.items():
            if key in self.accumulators:
                self.accumulators[key].merge(acc)
            else:
                self.accumulators[key] = copy.deepcopy(acc)

        self._cached_reduce.clear()
        return self

    def __repr__(self) -> str:
        """Return representation of the aggregator."""
        return (
            f'{self.__class__.__name__}(keys={list(self.accumulators.keys())})'
        )

    def clear(self) -> None:
        """Remove all accumulated data."""
        self.accumulators.clear()
        self._cached_reduce.clear()

    def keys(self) -> list[str]:
        """Return stored metric names."""
        return list(self.accumulators.keys())

    def reduce(self) -> dict[str, _R]:
        """Return reduced values for all metrics."""
        if not self._cached_reduce:
            self._cached_reduce = {
                key: acc.reduce() for key, acc in self.accumulators.items()
            }
        return self._cached_reduce

    def all_reduce(self) -> dict[str, _R]:
        """Synchronize accumulators across processes and reduce."""
        for acc in self.accumulators.values():
            acc.sync()

        self._cached_reduce.clear()
        return self.reduce()


@dataclass(slots=True)
class MeanAccumulator(AbstractAccumulator[float, float]):
    """Accumulator computing arithmetic mean for floats.

    Attributes:
        total: sum of all values.
        count: number of values.
    """

    total: float
    count: int

    @classmethod
    @override
    def from_value(cls, value: float) -> MeanAccumulator:
        return cls(total=value, count=1)

    @override
    def merge(self, other: Self) -> None:
        self.total += other.total
        self.count += other.count

    @override
    def reduce(self) -> float:
        return self.total / self.count

    @override
    def sync(self) -> None:
        return


@dataclass(slots=True)
class TorchMeanAccumulator(AbstractAccumulator[torch.Tensor, torch.Tensor]):
    """Accumulator computing arithmetic mean for tensors.

    Attributes:
        total: sum of all values.
        count: number of values.
    """

    total: torch.Tensor
    count: int

    @classmethod
    @override
    def from_value(cls, value: torch.Tensor) -> TorchMeanAccumulator:
        return cls(
            total=value.detach().sum(),
            count=value.numel(),
        )

    @override
    def merge(self, other: Self) -> None:
        self.total += other.total
        self.count += other.count

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


class Averager(AbstractAggregator[float, float]):
    """Aggregator computing mean over floats."""

    accumulator_cls = MeanAccumulator


class TorchAverager(AbstractAggregator[torch.Tensor, torch.Tensor]):
    """Aggregator computing mean over tensors with distributed support."""

    accumulator_cls = TorchMeanAccumulator

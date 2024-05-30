import collections
import itertools
from typing import Generic, TypeVar, Callable, Self, Hashable

import torch
from dry_torch import protocols
from dry_torch import data_types

_K = TypeVar('_K', bound=Hashable)
_V = TypeVar('_V')

_Output_contra = TypeVar('_Output_contra',
                         bound=data_types.OutputType,
                         contravariant=True)
_Target_contra = TypeVar('_Target_contra',
                         bound=data_types.TargetType,
                         contravariant=True)


class MetricsAggregate(
    collections.defaultdict[str, float],
    protocols.AggregateMapping
):

    def __add__(self, other: Self) -> Self:
        new_group = self.__class__()
        new_group.update(
            (key, self[key] + value) for key, value in other.items()
        )
        return new_group

    def __iadd__(self, other: protocols.AggregateMapping) -> Self:
        for key, value in other.items():
            self[key] += value
        return self


class Metrics(protocols.MetricsProtocol):
    """
    A base class to store batched values of the metrics_fun as Tensors.
    """
    __slots__ = 'metrics'

    def __init__(self, **metrics: torch.Tensor) -> None:
        self.metrics: MetricsAggregate = MetricsAggregate(float)
        renamed_metrics = map(_capitalize_keys, metrics.items())
        duplicated_metrics = itertools.tee(renamed_metrics)
        aggregated_metrics = map(_mean_value, duplicated_metrics[0])
        count_metrics = map(_get_sample_count, duplicated_metrics[1])
        self.metrics.update(itertools.chain(aggregated_metrics, count_metrics))
        self.metrics.update({})

    @staticmethod
    def reduce_metrics(metrics: protocols.AggregateMapping) -> dict[str, float]:
        counts: dict[str, float] = {}
        aggregates: dict[str, float] = {}
        for metric, value in metrics.items():
            if metric.startswith('count'):
                counts[metric.removeprefix('count_')] = value
            else:
                aggregates[metric] = value
        return {key: value / counts[key] for key, value in aggregates.items()}


class LossAndMetrics(Metrics, protocols.LossAndMetricsProtocol):
    """
    Stores the batched values of the loss_fun and other metrics_fun as torch 
    Tensors.
    """
    __slots__ = ('criterion', 'metrics')

    def __init__(self,
                 criterion: torch.Tensor,
                 **metrics: torch.Tensor) -> None:
        self.criterion = criterion.mean()
        self.metrics: MetricsAggregate
        super().__init__(criterion=criterion, **metrics)


class MetricsCalculator(Generic[_Output_contra, _Target_contra]):
    output_class: type[protocols.MetricsProtocol] = Metrics

    def __init__(
            self,
            **named_metric_fun: (
                    Callable[[_Output_contra, _Target_contra], torch.Tensor]
            ),
    ) -> None:
        self.named_metric_fun = named_metric_fun

    def __call__(self,
                 outputs: _Output_contra,
                 targets: _Target_contra) -> protocols.MetricsProtocol:
        metrics = self.output_class(**self._apply_fun(outputs, targets))
        return metrics

    def _apply_fun(self,
                   outputs: _Output_contra,
                   targets: _Target_contra) -> dict[str, torch.Tensor]:
        return {name: function(outputs, targets)
                for name, function in self.named_metric_fun.items()}


class LossAndMetricsCalculator(
    MetricsCalculator[_Output_contra, _Target_contra]
):

    def __init__(
            self,
            loss_fun: Callable[[_Output_contra, _Target_contra], torch.Tensor],
            **named_metric_fun: (
                    Callable[[_Output_contra, _Target_contra], torch.Tensor]
            ),
    ) -> None:
        self.loss_fun = loss_fun
        super().__init__(**named_metric_fun)

    def __call__(
            self,
            outputs: _Output_contra,
            targets: _Target_contra
    ) -> protocols.LossAndMetricsProtocol:
        return LossAndMetrics(
            criterion=self.loss_fun(outputs, targets),
            **self._apply_fun(outputs, targets)
        )


def _capitalize_keys(mapped_tuple: tuple[str, _V]) -> tuple[str, _V]:
    key = mapped_tuple[0]
    value = mapped_tuple[1]
    return key[0].upper() + key[1:], value


def _mean_value(mapped_tuple: tuple[_K, torch.Tensor]) -> tuple[_K, float]:
    key = mapped_tuple[0]
    value = mapped_tuple[1]
    return key, value.sum(0).item()


def _get_sample_count(mapped_tuple: tuple[str, _V]) -> tuple[str, int]:
    key = mapped_tuple[0]
    value = mapped_tuple[1]
    return 'count_' + key, len(value) if hasattr(value, '__len__') else 1

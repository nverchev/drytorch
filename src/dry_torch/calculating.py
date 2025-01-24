"""
Classes to create and combine loss and metrics.

The interface is similar to https://github.com/Lightning-AI/torchmetrics,
with stricter typing and slightly different constructions. One difference is
that torchmetrics MetricCollection and CompositionalMetric contain the
metrics they aggregate and can change their state, while here there is a more
functional approach.
"""

from __future__ import annotations

import abc
import warnings
from collections.abc import Hashable, Callable
from typing import TypeVar, Mapping, Self, Optional, Any, Protocol
from typing import runtime_checkable
from typing_extensions import override
import torch

from src.dry_torch import aggregators
from src.dry_torch import exceptions
from src.dry_torch import protocols as p

_K = TypeVar('_K', bound=Hashable)
_V = TypeVar('_V')

_Output_contra = TypeVar('_Output_contra',
                         bound=p.OutputType,
                         contravariant=True)
_Target_contra = TypeVar('_Target_contra',
                         bound=p.TargetType,
                         contravariant=True)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


@runtime_checkable
class TorchMetricCompositionalMetricProtocol(Protocol):
    """
    Protocol for a compositional metric from torchmetrics.

    Attributes:
        metric_a: first metric.
        metric_b: second metric.
    """
    metric_a: p.MetricCalculatorProtocol | float | None
    metric_b: p.MetricCalculatorProtocol | float | None

    def update(self,
               outputs: torch.Tensor,
               targets: torch.Tensor) -> Any:
        """See torchmetrics documentation."""

    def reset(self) -> Any:
        """See torchmetrics documentation."""

    def compute(self) -> Mapping[str, torch.Tensor] | torch.Tensor | None:
        """See torchmetrics documentation."""

    def forward(self,
                outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """See torchmetrics documentation."""

    def __call__(self,
                 outputs: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        """See torchmetrics documentation."""


def from_torchmetrics(
        metric: TorchMetricCompositionalMetricProtocol
) -> p.LossCalculatorProtocol[torch.Tensor, torch.Tensor]:
    """Wrapper of a CompositionalMetric for integration."""

    class TorchMetricCompositionalMetric(
        p.LossCalculatorProtocol[torch.Tensor, torch.Tensor]
    ):
        name = 'Loss'

        def __init__(self, _metric: TorchMetricCompositionalMetricProtocol):
            self.metric = _metric

        def update(self,
                   outputs: torch.Tensor,
                   targets: torch.Tensor) -> Any:
            self.metric.update(outputs, targets)

        def reset(self) -> Any:
            self.metric.reset()

        def forward(self,
                    outputs: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
            return self.metric(outputs, targets)

        def compute(self) -> dict[str, torch.Tensor]:
            dict_output = dict[str, torch.Tensor]()
            metric_list = list[p.MetricCalculatorProtocol | float | None]()
            metric_list.append(self.metric)

            while metric_list:
                metric_ = metric_list.pop()
                if isinstance(metric_, self.metric.__class__):
                    metric_list.extend([metric_.metric_b, metric_.metric_a])
                elif isinstance(metric_, (float, int)) or metric_ is None:
                    continue
                else:
                    if isinstance(value := metric_.compute(), torch.Tensor):
                        dict_output[metric_.__class__.__name__] = value
            return dict_output

    return TorchMetricCompositionalMetric(metric)


class MetricBase(
    p.MetricCalculatorProtocol[_Output_contra, _Target_contra],
    metaclass=abc.ABCMeta
):
    def __init__(self,
                 **metric_fun: p.TensorCallable[_Output_contra, _Target_contra],
                 ) -> None:
        self._aggregator = aggregators.TorchAverager()
        self.named_metric_fun = metric_fun

    @override
    def compute(self: Self) -> dict[str, torch.Tensor]:
        """Return a Mapping from the metric name to the calculated value."""
        if not self._aggregator:
            warnings.warn(exceptions.ComputedBeforeUpdatedWarning(self))
        return self._aggregator.reduce()

    @override
    def update(self: Self,
               outputs: _Output_contra,
               targets: _Target_contra) -> dict[str, torch.Tensor]:
        results = self.calculate(outputs, targets)
        self._aggregator += results
        return results

    @override
    def reset(self: Self) -> None:
        """Reset cached values."""
        self._aggregator.clear()
        return

    @abc.abstractmethod
    def calculate(self: Self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, torch.Tensor]:
        """
        Actual method responsible for the calculations.

        Args:
            outputs: model outputs.
            targets: ground truth.
        """

    def __or__(
            self,
            other: MetricBase[_Output_contra, _Target_contra]
    ) -> MetricCollection[_Output_contra, _Target_contra]:
        named_metric_fun = self.named_metric_fun | other.named_metric_fun
        return MetricCollection(**named_metric_fun)


class MetricCollection(MetricBase[_Output_contra, _Target_contra]):

    @override
    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, torch.Tensor]:
        return dict_apply(self.named_metric_fun, outputs, targets)


class Metric(MetricBase[_Output_contra, _Target_contra]):

    def __init__(self,
                 name: str,
                 fun: p.TensorCallable[_Output_contra, _Target_contra],
                 higher_is_better: Optional[bool] = None) -> None:
        super().__init__(**{name: fun})
        self.name = name
        self.fun = fun
        self.higher_is_better = higher_is_better

    @override
    def calculate(self: Self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, torch.Tensor]:
        return {self.name: self.fun(outputs, targets)}


class LossBase(
    MetricBase[_Output_contra, _Target_contra],
    p.LossCalculatorProtocol[_Output_contra, _Target_contra],
    metaclass=abc.ABCMeta,
):
    name = 'Loss'

    def __init__(
            self,
            criterion: Callable[[dict[str, torch.Tensor]], torch.Tensor],
            **named_metric_fun: p.TensorCallable[_Output_contra, _Target_contra]
    ) -> None:
        super().__init__(**named_metric_fun)
        self.criterion = criterion
        return

    @override
    def forward(self,
                outputs: _Output_contra,
                targets: _Target_contra) -> torch.Tensor:
        metrics = self.update(outputs, targets)
        return self.criterion(metrics)

    def __or__(
            self,
            other: MetricBase[_Output_contra, _Target_contra]
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        named_metric_fun = self.named_metric_fun | other.named_metric_fun
        return CompositionalLoss(self.criterion, **named_metric_fun)

    def _apply(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
            operation: Callable[
                [torch.Tensor, torch.Tensor | float],
                torch.Tensor
            ],
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """
        Helper method to combine two losses or apply an operation with a float.
        """
        if isinstance(other, LossBase):
            named_metric_fun = self.named_metric_fun | other.named_metric_fun

            def _new_criterion(x: dict[str, torch.Tensor]) -> torch.Tensor:
                return operation(self.criterion(x), other.criterion(x))

        elif isinstance(other, (float, int)):
            named_metric_fun = self.named_metric_fun

            def _new_criterion(x: dict[str, torch.Tensor]) -> torch.Tensor:
                return operation(self.criterion(x), other)
        else:
            raise TypeError(f"Unsupported type for operation: {type(other)}")

        return CompositionalLoss(_new_criterion, **named_metric_fun)

    def __add__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        return self._apply(other, lambda t1, t2: t1 + t2)

    def __radd__(self, other: float) -> CompositionalLoss:
        return self.__add__(other)

    def __sub__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        return self._apply(other, lambda t1, t2: t1 - t2)

    def __rsub__(self, other: float) -> CompositionalLoss:
        return self._apply(other, lambda t1, t2: t2 - t1)

    def __mul__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        return self._apply(other, lambda t1, t2: t1 * t2)

    def __rmul__(self, other: float) -> CompositionalLoss:
        return self.__mul__(other)

    def __truediv__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        return self._apply(other, lambda t1, t2: t1 / t2)

    def __rtruediv__(self, other: float) -> CompositionalLoss:
        return self._apply(other, lambda t1, t2: t2 / t1)

    def __neg__(self) -> CompositionalLoss:
        return CompositionalLoss(lambda x: -self.criterion(x),
                                 **self.named_metric_fun)


class CompositionalLoss(
    LossBase[_Output_contra, _Target_contra],
    MetricCollection[_Output_contra, _Target_contra]
):

    def __init__(
            self,
            criterion: Callable[[dict[str, torch.Tensor]], torch.Tensor],
            **named_metric_fun: p.TensorCallable[_Output_contra, _Target_contra]
    ) -> None:
        super().__init__(criterion, **named_metric_fun)
        return

    def calculate(self: Self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, torch.Tensor]:
        all_metrics = super().calculate(outputs, targets)
        return {self.name: self.criterion(all_metrics)} | all_metrics


class Loss(
    LossBase[_Output_contra, _Target_contra],
    Metric[_Output_contra, _Target_contra],
):

    def __init__(
            self,
            name: str,
            fun: p.TensorCallable[_Output_contra, _Target_contra],
            higher_is_better: Optional[bool] = False):
        def _criterion(all_metrics: dict[str, torch.Tensor]) -> torch.Tensor:
            return all_metrics[name]

        super(LossBase, self).__init__(name, fun, higher_is_better)
        self.criterion = _criterion
        return


def dict_apply(
        dict_fun: dict[str, p.TensorCallable[_Output_contra, _Target_contra]],
        outputs: _Output_contra,
        targets: _Target_contra,
) -> dict[str, torch.Tensor]:
    """
    Apply the given tensor callables to the provided outputs and targets.

    Args:
        dict_fun: A dictionary of named callables (outputs, targets) -> Tensor.
        outputs: The outputs to apply the tensor callables to.
        targets: The targets to apply the tensor callables to.

    Returns:
        A dictionary containing the resulting values.
    """
    return {name: function(outputs, targets)
            for name, function in dict_fun.items()}


def repr_metrics(calculator: p.MetricCalculatorProtocol) -> Mapping[str, float]:
    """Represents the metrics as a mapping of named values."""
    metrics = calculator.compute()
    if isinstance(metrics, Mapping):
        return {name: value.item() for name, value in metrics.items()}
    if isinstance(metrics, torch.Tensor):
        return {calculator.__class__.__name__: metrics.item()}
    return {}

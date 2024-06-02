from typing import Generic, TypeVar, Callable, Hashable

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


class Metrics(protocols.MetricsProtocol):
    """
    A base class to store batched values of the metrics_fun as Tensors.
    """
    __slots__ = 'metrics'

    def __init__(self, **metrics: torch.Tensor) -> None:
        self.metrics = metrics


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
        super().__init__(criterion=criterion, **metrics)


class MetricsCalculator(Generic[_Output_contra, _Target_contra]):

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
        metrics = Metrics(**self._apply_fun(outputs, targets))
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
        out = LossAndMetrics(
            criterion=self.loss_fun(outputs, targets),
            **self._apply_fun(outputs, targets),
        )
        return out

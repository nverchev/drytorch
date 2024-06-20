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
_Target = TypeVar('_Target', bound=data_types.TargetType)
_Output = TypeVar('_Output', bound=data_types.OutputType)


class LossAndMetrics(protocols.LossAndMetricsProtocol):
    """
    Stores the values of the average loss and batched metrics as torch Tensors.
    """
    __slots__ = ('criterion', 'metrics')

    def __init__(self,
                 criterion: torch.Tensor,
                 **metrics: torch.Tensor) -> None:
        self.criterion = criterion.mean()
        self.metrics = dict(criterion=criterion) | metrics


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
                 targets: _Target_contra) -> dict[str, torch.Tensor]:
        return {name: function(outputs, targets)
                for name, function in self.named_metric_fun.items()}


class LossCalculator(Generic[_Output, _Target]):

    def __init__(
            self,
            loss_fun: Callable[[_Output, _Target], torch.Tensor],
            **named_metric_fun: Callable[[_Output, _Target], torch.Tensor],
    ) -> None:
        self.loss_fun = loss_fun
        self.metrics_calc: protocols.MetricsCallable[_Output, _Target] = MetricsCalculator(**named_metric_fun)

    def __call__(
            self,
            outputs: _Output,
            targets: _Target
    ) -> protocols.LossAndMetricsProtocol:
        out = LossAndMetrics(
            criterion=self.loss_fun(outputs, targets),
            **self.metrics_calc(outputs, targets)
        )
        return out

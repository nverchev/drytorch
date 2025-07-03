"""Module containing utilies to ensure compatibility with torchmetrics."""

from __future__ import annotations

from typing import runtime_checkable, Protocol, Any, Mapping

from drytorch import protocols as p
from drytorch.metrics import _Tensor


@runtime_checkable
class TorchMetricCompositionalMetricProtocol(Protocol):
    """
    Protocol for a compositional metric from torchmetrics.

    Attributes:
        metric_a: first metric.
        metric_b: second metric.
    """
    metric_a: p.ObjectiveProtocol | float | None
    metric_b: p.ObjectiveProtocol | float | None

    def update(self,
               outputs: _Tensor,
               targets: _Tensor) -> Any:
        """See torchmetrics documentation."""

    def reset(self) -> Any:
        """See torchmetrics documentation."""

    def compute(self) -> Mapping[str, _Tensor] | _Tensor | None:
        """See torchmetrics documentation."""

    def forward(self,
                outputs: _Tensor,
                targets: _Tensor) -> _Tensor:
        """See torchmetrics documentation."""

    def __call__(self,
                 outputs: _Tensor,
                 targets: _Tensor) -> _Tensor:
        """See torchmetrics documentation."""


def from_torchmetrics(
        metric: TorchMetricCompositionalMetricProtocol
) -> p.LossCalculatorProtocol[_Tensor, _Tensor]:
    """Returns a wrapper of a CompositionalMetric for integration."""

    class _TorchMetricCompositionalMetric(
        p.LossCalculatorProtocol[_Tensor, _Tensor]
    ):
        name = 'Loss'

        def __init__(self, _metric: TorchMetricCompositionalMetricProtocol):
            self.metric = _metric

        def update(self,
                   outputs: _Tensor,
                   targets: _Tensor) -> Any:
            self.metric.update(outputs, targets)

        def reset(self) -> Any:
            self.metric.reset()

        def forward(self,
                    outputs: _Tensor,
                    targets: _Tensor) -> _Tensor:
            return self.metric(outputs, targets)

        def compute(self) -> dict[str, _Tensor]:
            dict_output = dict[str, _Tensor]()
            metric_list = list[p.ObjectiveProtocol | float | None]()
            metric_list.append(self.metric)
            while metric_list:
                metric_ = metric_list.pop()
                if isinstance(metric_, self.metric.__class__):
                    metric_list.extend([metric_.metric_b, metric_.metric_a])
                elif isinstance(metric_, (float, int)) or metric_ is None:
                    continue
                else:
                    if isinstance(value := metric_.compute(), _Tensor):
                        dict_output[metric_.__class__.__name__] = value

            return dict_output

    return _TorchMetricCompositionalMetric(metric)

"""Module containing utilies to ensure compatibility with torchmetrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from torchmetrics import metric

from drytorch.core import protocols as p


if TYPE_CHECKING:
    from torchmetrics import metric


_Tensor = torch.Tensor


def from_torchmetrics(
    torch_metric: metric.CompositionalMetric,
) -> p.LossProtocol[_Tensor, _Tensor]:
    """Returns a wrapper of a CompositionalMetric for integration."""

    class _TorchMetricCompositionalMetric(p.LossProtocol[_Tensor, _Tensor]):
        name = 'Loss'

        def __init__(self, _metric: metric.CompositionalMetric) -> None:
            self.metric = _metric

        def update(self, outputs: _Tensor, targets: _Tensor) -> Any:
            self.metric.update(outputs, targets)

        def reset(self) -> Any:
            self.metric.reset()

        def forward(self, outputs: _Tensor, targets: _Tensor) -> _Tensor:
            return self.metric(outputs, targets)

        def compute(self) -> dict[str, _Tensor]:
            dict_output = dict[str, _Tensor]()
            metric_list = list[type(self.metric.metric_b)]()
            metric_list.append(self.metric)
            while metric_list:
                metric_ = metric_list.pop()
                if isinstance(metric_, self.metric.__class__):
                    metric_list.extend([metric_.metric_b, metric_.metric_a])
                elif (
                    isinstance(metric_, float | int | _Tensor)
                    or metric_ is None
                ):
                    continue
                else:
                    if isinstance(value := metric_.compute(), _Tensor):
                        dict_output[metric_.__class__.__name__] = value

            return dict_output

    return _TorchMetricCompositionalMetric(torch_metric)

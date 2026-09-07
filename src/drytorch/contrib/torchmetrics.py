"""Module containing utilies to ensure compatibility with torchmetrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from drytorch.core import exceptions
from drytorch.core import protocols as p


__all__ = [
    'from_torchmetrics',
]

if TYPE_CHECKING:
    from torchmetrics import metric

_Tensor = torch.Tensor


def from_torchmetrics(
    torch_metric: metric.CompositionalMetric,
) -> p.LossProtocol[_Tensor, _Tensor]:
    """Returns a wrapper of a CompositionalMetric for integration.

    Raises:
        RepeatedMetricsError: if two components of the same class are
            present in the composition. Subclass the metric so each
            configuration has its own class name.
    """

    class _TorchMetricCompositionalMetric(p.LossProtocol[_Tensor, _Tensor]):
        """Wrapper of Compositional Metric reporting internal calculations.

        Note that this CompositionalMetric can be used as loss.
        """

        name = 'Loss'

        def __init__(self, _metric: metric.CompositionalMetric) -> None:
            self.metric = _metric
            self.metric.sync_on_compute = False
            self.metric.dist_sync_on_step = False
            self._leaf_metrics = _get_leaf_metrics(self.metric)
            _disable_leaf_syncing(self._leaf_metrics)
            return

        def compute(self) -> dict[str, _Tensor]:
            """Output a dictionary of metric values for each component."""
            dict_output = dict[str, _Tensor](
                {'Combined Loss': self.metric.compute()}
            )
            for m in self._leaf_metrics:
                if isinstance(value := m.compute(), _Tensor):
                    dict_output[m.__class__.__name__] = value

            return dict_output

        def forward(self, outputs: _Tensor, targets: _Tensor) -> _Tensor:
            return self.metric(outputs, targets)

        def reset(self) -> Any:
            self.metric.reset()

        def update(self, outputs: _Tensor, targets: _Tensor) -> Any:
            self.metric.update(outputs, targets)

    return _TorchMetricCompositionalMetric(torch_metric)


def _get_leaf_metrics(
    comp_metric: metric.CompositionalMetric,
) -> list[metric.Metric]:
    leaves: list[metric.Metric] = []
    stack: list[metric.Metric | float | int | _Tensor | None] = [comp_metric]
    names_seen = set[str]()
    while stack:
        node = stack.pop()
        if isinstance(node, comp_metric.__class__):
            stack.extend([node.metric_b, node.metric_a])
        elif isinstance(node, float | int | _Tensor) or node is None:
            continue
        else:
            leaves.append(node)
            name = node.__class__.__name__
            if name in names_seen:
                raise exceptions.RepeatedMetricsError([name])

            names_seen.add(name)

    return leaves


def _disable_leaf_syncing(leaf_metrics: list[metric.Metric]) -> None:
    for m in leaf_metrics:
        m.sync_on_compute = False
        m.dist_sync_on_step = False

    return

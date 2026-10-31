"""Utilities to ensure compatibility with torchmetrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

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
    /,
) -> p.SyncedLossProtocol[_Tensor, _Tensor]:
    """Returns a wrapper of a CompositionalMetric for integration.

    Args:
        torch_metric: the composed torchmetrics metric to wrap.

    Returns:
        The wrapped metric, usable as a loss.

    Raises:
        RepeatedMetricsError: if two components of the same class are
            present in the composition. Subclass the metric so each
            configuration has its own class name.
    """

    class _TorchMetricCompositionalMetric:
        """Wrapper of Compositional Metric reporting internal calculations.

        Note that this CompositionalMetric can be used as loss.
        """

        higher_is_better: Final[bool | None]
        metric: Final[metric.CompositionalMetric]
        name = 'Loss'
        _leaf_metrics: list[metric.Metric]

        def __init__(self, _metric: metric.CompositionalMetric) -> None:
            self.higher_is_better = _metric.higher_is_better
            self.metric = _metric
            self.metric.sync_on_compute = False
            self.metric.dist_sync_on_step = False
            self._leaf_metrics = _get_leaf_metrics(self.metric)
            _disable_leaf_syncing(self._leaf_metrics)
            return

        def compute(self) -> dict[str, _Tensor]:
            """Output a dictionary of metric values for each component."""
            dict_output = dict[str, _Tensor]({self.name: self.metric.compute()})
            for m in self._leaf_metrics:
                if isinstance(value := m.compute(), _Tensor):
                    dict_output[m.__class__.__name__] = value

            return dict_output

        def forward(self, outputs: _Tensor, targets: _Tensor) -> _Tensor:
            return self.metric(outputs, targets)

        def reset(self) -> None:
            self.metric.reset()
            return

        def sync(self) -> None:
            for m in self._leaf_metrics:
                m.sync()

            return

        def update(self, outputs: _Tensor, targets: _Tensor) -> None:
            self.metric.update(outputs, targets)
            return

    return _TorchMetricCompositionalMetric(torch_metric)


def _get_leaf_metrics(
    comp_metric: metric.CompositionalMetric,
) -> list[metric.Metric]:
    leaves: list[metric.Metric] = []
    stack: list[metric.Metric | float | int | _Tensor | None] = [comp_metric]
    names_seen: set[str] = set()
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

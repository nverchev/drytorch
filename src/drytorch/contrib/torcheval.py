"""Support for torcheval syncing recommended for distributed training."""

from __future__ import annotations

from typing import Final

import torch

from torcheval import metrics
from torcheval.metrics import toolkit

from drytorch.core import exceptions
from drytorch.core import protocols as p


__all__ = [
    'from_torcheval',
]


_Tensor = torch.Tensor


def from_torcheval(
    torch_eval: metrics.Metric[_Tensor | dict[str, _Tensor]],
    name: str | None = None,
) -> p.ObjectiveProtocol[_Tensor, _Tensor]:
    """Returns a wrapper of a Metric from torcheval with a sync method.

    Args:
        torch_eval: the torcheval metric to wrap.
        name: optional name for the metric. Defaults to the class name.

    Raises:
        exceptions.ComputedMetricsTypeError: If the metric computes to an
            invalid type.
    """

    class _TorchEvalWithSync:
        metric: Final[metrics.Metric[_Tensor | dict[str, _Tensor]]]
        name: Final[str]
        _synced_value: _Tensor | dict[str, _Tensor] | None

        def __init__(
            self, _metric: metrics.Metric[_Tensor | dict[str, _Tensor]]
        ) -> None:
            self.metric = _metric
            name_str = name if name is not None else _metric.__class__.__name__
            self.name = name_str
            self._synced_value = None
            return

        def compute(self) -> dict[str, _Tensor]:
            if self._synced_value is None:
                val = self.metric.compute()
            else:
                val = self._synced_value

            if isinstance(val, dict):
                return val

            if not isinstance(val, _Tensor):
                raise exceptions.ComputedMetricsTypeError(type(val))

            return {self.name: val}

        def reset(self) -> None:
            self.metric.reset()
            self._synced_value = None
            return

        def sync(self) -> None:
            """Use torcheval toolkit to synchronize and compute metrics."""
            self._synced_value = toolkit.sync_and_compute(self.metric)
            return

        def update(self, outputs: _Tensor, targets: _Tensor) -> None:
            self.metric.update(outputs, targets)
            self._synced_value = None
            return

    return _TorchEvalWithSync(torch_eval)

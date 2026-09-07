"""Support for torcheval syncing recommended for distributed training."""

import torch

from torcheval import metrics
from torcheval.metrics import toolkit

from drytorch.core import protocols as p


_Tensor = torch.Tensor


def from_torcheval(
    torch_eval: metrics.Metric[_Tensor | dict[str, _Tensor]],
    name: str | None = None,
) -> p.ObjectiveProtocol[_Tensor, _Tensor]:
    """Returns a wrapper of a Metric from torcheval with a sync method."""

    class _TorchEvalWithSync(p.ObjectiveProtocol[_Tensor, _Tensor]):
        def __init__(
            self, _metric: metrics.Metric[_Tensor | dict[str, _Tensor]]
        ) -> None:
            self.metric = _metric
            self.name = name if name is not None else _metric.__class__.__name__
            self._synced_value: _Tensor | dict[str, _Tensor] | None = None
            self._check_return_type()
            return

        def _check_return_type(self) -> None:
            """Check the return type of the metric."""
            initial_val = self.metric.compute()
            if not isinstance(initial_val, (torch.Tensor, dict)):
                raise TypeError(
                    'torcheval metric must return a Tensor or dict of Tensors'
                )

            return

        def compute(self) -> dict[str, _Tensor]:
            if self._synced_value is not None:
                val = self._synced_value
            else:
                val = self.metric.compute()

            if isinstance(val, dict):
                return val

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

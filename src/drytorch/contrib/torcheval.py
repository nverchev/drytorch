"""Support for torcheval syncing recommended for distributed training."""

import torch

from torcheval import metrics
from torcheval.metrics import toolkit

from drytorch.core import protocols as p


_Tensor = torch.Tensor


def from_torcheval(
    torch_eval: metrics.Metric[_Tensor | dict[str, _Tensor]],
) -> p.ObjectiveProtocol[_Tensor, _Tensor]:
    """Returns a wrapper of a Metric from torcheval with a sync method."""

    class _TorchEvalWithSync(p.ObjectiveProtocol[_Tensor, _Tensor]):
        name = 'Loss'

        def __init__(
            self, _metric: metrics.Metric[_Tensor | dict[str, _Tensor]]
        ) -> None:
            self.metric = _metric
            return

        def compute(self) -> dict[str, _Tensor]:
            output = self.metric.compute()
            if isinstance(output, dict):
                return output

            return {self.metric.__name__: output}

        def reset(self) -> None:
            self.metric.reset()
            return

        def sync(self) -> None:
            """Use torcheval toolkit to synchronize and compute metrics."""
            toolkit.sync_and_compute(self.metric)
            return

        def update(self, outputs: _Tensor, targets: _Tensor) -> None:
            self.metric.update(outputs, targets)
            return

    return _TorchEvalWithSync(torch_eval)

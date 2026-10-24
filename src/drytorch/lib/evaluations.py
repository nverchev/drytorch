"""Classes for the evaluation of a model."""

from collections.abc import Mapping
from typing import Any, TypeVar

import torch

from typing_extensions import override

from drytorch.core import log_events
from drytorch.core import protocols as p
from drytorch.lib import runners


__all__ = [
    'Evaluation',
    'Test',
    'Validation',
]


Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)


class Evaluation(runners.Source[Input, Target, Output, Any]):
    """Evaluate a model in inference mode.

    Attributes:
        loader: provides inputs and targets in batches.
        model: the model containing the weights to evaluate.
        objective: processes the model outputs and targets.
    """

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        *,
        loader: p.LoaderProtocol[tuple[Input, Target]],
        name: str = '',
        metric: p.ObjectiveProtocol[Output, Target],
    ) -> None:
        """Initialize.

        Args:
            model: the model containing the weights to evaluate.
            loader: provides inputs and targets in batches.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            metric: metric to evaluate the model.

        """
        super().__init__(model, loader=loader, name=name, objective=metric)
        return

    @override
    @torch.inference_mode()
    def __call__(self, *, store_outputs: bool = False) -> None:
        """Set the model in evaluation mode and PyTorch in inference mode."""
        self.model.module.eval()
        super().__call__(store_outputs=store_outputs)
        return


class Validation(Evaluation[Input, Target, Output]):
    """Evaluate model performance on held-out data during training.

    Attributes:
        loader: provides inputs and targets in batches.
        model: the model containing the weights to evaluate.
        objective: processes the model outputs and targets.
    """


class Test(Evaluation[Input, Target, Output]):
    """Evaluate a model outside the training loop.

    Attributes:
        loader: provides inputs and targets in batches.
        model: the model containing the weights to evaluate.
        objective: processes the model outputs and targets.
    """

    @override
    def _publish_metrics(self, metrics: Mapping[str, float]) -> None:
        log_events.TestMetricEvent(
            model_name=self.model.name,
            source_name=self.name,
            epoch=self.model.epoch,
            metrics=metrics,
        )
        return

"""Module containing classes for the evaluation of a model."""

from typing import Any, Protocol, TypeVar

import torch

from typing_extensions import override

from drytorch.core import log_events
from drytorch.core import protocols as p
from drytorch.lib import runners


__all__ = [
    'Diagnostic',
    'EvaluationMixin',
    'Test',
    'Validation',
]


Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)


class _RunnerLike(Protocol):
    model: p.ModelProtocol[Any, Any]

    def __call__(self, store_outputs: bool = False) -> None: ...


class EvaluationMixin:
    """Mixin for running inference in eval mode without gradients."""

    @torch.inference_mode()
    def __call__(
        self: _RunnerLike,
        store_outputs: bool = False,
    ) -> None:
        """Set the model in evaluation mode and PyTorch in inference mode."""
        self.model.module.eval()
        super().__call__(store_outputs)  # type: ignore
        return


class Diagnostic(
    EvaluationMixin,
    runners.ModelRunnerWithLogs[Input, Target, Output, Any],
):
    """Evaluate the model on inference mode without logging the metrics.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        objective: processes the model outputs and targets.
        outputs_list: list of optionally stored outputs.
    """


class Validation(
    EvaluationMixin,
    runners.ModelRunnerWithLogs[Input, Target, Output, Any],
):
    """Evaluate model on inference mode.

    It could be used for testing (see subclass) or validating a model.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        objective: processes the model outputs and targets.
        outputs_list: list of optionally stored outputs.
    """

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        name: str = '',
        *,
        loader: p.LoaderProtocol[tuple[Input, Target]],
        metric: p.ObjectiveProtocol[Output, Target],
    ) -> None:
        """Constructor.

        Args:
            model: the model containing the weights to evaluate.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            loader: provides inputs and targets in batches.
            metric: metric to evaluate the model.

        """
        super().__init__(model, loader=loader, name=name, objective=metric)
        return


class Test(Validation[Input, Target, Output]):
    """Evaluate model performance on a test dataset.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        objective: processes the model outputs and targets.
        outputs_list: list of optionally stored outputs.
    """

    @override
    def __call__(self, store_outputs: bool = False) -> None:
        """Test the model on the dataset.

        Args:
            store_outputs: whether to store model outputs. Defaults to False.
        """
        log_events.StartTestEvent(self.name, self.model.name)
        super().__call__(store_outputs)
        log_events.EndTestEvent(self.name, self.model.name)
        return

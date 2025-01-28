"""Classes for the evaluation of a model."""
import sys
import abc
from typing import Any, Mapping, TypeVar
import warnings

from typing_extensions import override

import torch

from src.dry_torch import apply_ops
from src.dry_torch import calculating
from src.dry_torch import exceptions
from src.dry_torch import loading
from src.dry_torch import log_events
from src.dry_torch import protocols as p
from src.dry_torch import registering
from src.dry_torch import repr_utils

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


class Evaluation(p.EvaluationProtocol[_Input, _Target, _Output]):
    """
    Abstract class for evaluating a model on a given dataset.

    It coordinates the batching from a loader with the processing of the
    model output.
    Subclasses need to implement the __call__ method for training, validation or
    testing of the model.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        calculator: processes the model outputs and targets.
        name: the name for the object for logging purposes.
        mixed_precision: whether to use mixed precision computing.
        outputs_list: list of optionally stored outputs
    """
    max_stored_output: int = sys.maxsize
    _default_name = repr_utils.DefaultName()

    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            calculator: p.MetricCalculatorProtocol[_Output, _Target],
            name: str = '',
            mixed_precision: bool = False,
    ) -> None:
        """
        Args:
            model: the model containing the weights to evaluate.
            loader: provides inputs and targets in batches.
            calculator: processes the model outputs and targets.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            mixed_precision: whether to use mixed precision computing.
                Defaults to False.
        """
        self.model = model
        self.name = repr_utils.StrWithTS(name or self._default_name)
        self.loader = loader
        self.calculator = calculator
        device_is_cuda = self.model.device.type == 'cuda'
        self.mixed_precision = mixed_precision and device_is_cuda
        self.outputs_list = list[_Output]()
        self._metadata_recorded = False
        return

    @abc.abstractmethod
    def __call__(self) -> None:
        """
        Abstract method to be implemented by subclasses for model evaluation.
        """
        registering.record_model_call(self, self.model)
        return

    def _log_metrics(self, metrics: Mapping[str, Any]) -> None:
        log_events.FinalMetrics(model_name=self.model.name,
                                source=self.name,
                                epoch=self.model.epoch,
                                metrics=metrics)
        return

    def _run_backwards(self, outputs: _Output, targets: _Target) -> None:
        self.calculator.update(outputs, targets)

    def _run_forward(self, inputs: _Input) -> _Output:
        with torch.autocast(device_type=self.model.device.type,
                            enabled=self.mixed_precision):
            return self.model(inputs)

    def _run_epoch(self, store_outputs: bool):
        self.outputs_list.clear()
        self.calculator.reset()
        num_samples = loading.check_dataset_length(self.loader.dataset)
        pbar = log_events.IterateBatch(self.name, len(self.loader), num_samples)
        metrics: Mapping[str, Any] = {}
        for batch in self.loader:
            inputs, targets = apply_ops.apply_to(batch, self.model.device)
            outputs = self._run_forward(inputs)
            self._run_backwards(outputs, targets)
            pbar.update(calculating.repr_metrics(self.calculator))
            if store_outputs:
                self._store(outputs)

        self._log_metrics(metrics)

    def _store(self, outputs: _Output) -> None:
        try:
            outputs = apply_ops.apply_cpu_detach(outputs)
        except (exceptions.FuncNotApplicableError,
                exceptions.NamedTupleOnlyError) as err:
            warnings.warn(exceptions.CannotStoreOutputWarning(err))
        else:
            self.outputs_list.append(outputs)

    def __repr__(self) -> str:
        return self.name + f'for model {self.model.name}'

    def __str__(self) -> str:
        return str(self.name)


class Diagnostic(Evaluation[_Input, _Target, _Output]):
    """
    Evaluate model on inference mode.

    It could be used for testing or validating a model (see subclasses) but
    also for diagnosing a problem in its training.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        calculator: processes the model outputs and targets.
        name: the name for the object for logging purposes.
        mixed_precision: whether to use mixed precision computing.
        outputs_list: list of optionally stored outputs
    """

    @override
    @torch.inference_mode()
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Run epoch without tracking gradients and in eval mode.

        Args:
            store_outputs: whether to store model outputs. Defaults to False
        """
        super().__call__()
        self.model.module.eval()
        self._run_epoch(store_outputs)
        return


class Validation(Diagnostic[_Input, _Target, _Output]):
    """
    Evaluate model performance on a validation dataset.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        calculator: processes the model outputs and targets.
        name: the name for the object for logging purposes.
        mixed_precision: whether to use mixed precision computing.
        outputs_list: list of optionally stored outputs
    """


class Test(Diagnostic[_Input, _Target, _Output]):
    """
    Evaluate model performance on a test dataset.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        calculator: processes the model outputs and targets.
        name: the name for the object for logging purposes.
        mixed_precision: whether to use mixed precision computing.
        outputs_list: list of optionally stored outputs
    """

    @override
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Test the model on the dataset.

        Args:
            store_outputs: whether to store model outputs. Defaults to False
        """
        log_events.Test(self.model.name, self.name)
        super().__call__(store_outputs)
        return

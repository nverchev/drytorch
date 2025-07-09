"""Module containing classes that run a model."""

import copy
import sys
from typing import Iterator, Mapping, Generic, TypeVar
from typing_extensions import override
import warnings

from drytorch import exceptions
from drytorch import loading
from drytorch import log_events
from drytorch import metrics
from drytorch import protocols as p
from drytorch import registering
from drytorch.utils import apply_ops
from drytorch.utils import repr_utils

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


class ModelCaller(Generic[_Input, _Output]):
    """
    Base class that calls a model.

    Attributes:
        model: the wrapped model.
    """
    _name = repr_utils.DefaultName()

    def __init__(self,
                 model: p.ModelProtocol[_Input, _Output],
                 name: str = '') -> None:
        """
        Args:
            model: the wrapped model.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
        """
        self.model = model
        self._name = name

    def __call__(self):
        pass

    def __repr__(self) -> str:
        return f'{self.name}({self.model.name})'

    @property
    def name(self) -> str:
        """The name of the object calling the model."""
        return self._name


class Source(ModelCaller[_Input, _Output], repr_utils.Versioned):
    """
    Document itself when the model is first called.

    Attributes:
        model: the model containing the weights to evaluate.
    """

    def __init__(self,
                 model: p.ModelProtocol[_Input, _Output],
                 name: str = ''):
        """
        Args:
            model: the model containing the weights to evaluate.
        """
        super().__init__(model, name)
        self._registered = False
        return

    def __call__(self) -> None:
        """Record call."""
        if not self._registered:
            registering.record_model_call(self, self.model)

        self._registered = True
        super().__call__()
        return


class ModelRunner(ModelCaller, Generic[_Input, _Target, _Output]):
    """
    Run a model on a dataset.

    Attributes:
        model: the model to run.
        loader: the loader providing inputs and targets in batches.
        outputs_list: list of optionally stored outputs.
    """
    max_stored_output: int = sys.maxsize

    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            name: str = '',
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
    ) -> None:
        """
        Args:
            model: the model to run.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            loader: provides inputs and targets in batches.

        """
        super().__init__(model, name)
        self.model = model
        self.loader = loader
        self.outputs_list = list[_Output]()
        return

    def __call__(self, store_outputs: bool = False) -> None:
        """
        Single pass on the dataset.

        Args:
            store_outputs: whether to store model outputs. Defaults to False.
        """
        super().__call__()
        self._run_epoch(store_outputs)
        return

    def _get_batches(self) -> Iterator[tuple[_Input, _Target]]:
        return (apply_ops.apply_to(batch, self.model.device)
                for batch in self.loader)

    def _get_metrics(self) -> Mapping[str, float]:
        return {}

    def _run_backward(self, outputs: _Output, targets: _Target) -> None:
        return

    def _run_batch(self, batch: tuple[_Input, _Target], ) -> _Output:
        inputs, targets = batch
        outputs = self._run_forward(inputs)
        self._run_backward(outputs, targets)
        return outputs

    def _run_epoch(self, store_outputs: bool):
        self.outputs_list.clear()
        num_samples = loading.check_dataset_length(self.loader.dataset)
        pbar = log_events.IterateBatch(self.name,
                                       self.loader.batch_size,
                                       len(self.loader),
                                       num_samples)
        for batch in self._get_batches():
            outputs = self._run_batch(batch)
            pbar.update(self._get_metrics())
            if store_outputs:
                self._store(outputs)

    def _run_forward(self, inputs: _Input) -> _Output:
        return self.model(inputs)

    def _store(self, outputs: _Output) -> None:
        try:
            outputs = apply_ops.apply_cpu_detach(outputs)
        except (exceptions.FuncNotApplicableError,
                exceptions.NamedTupleOnlyError) as err:
            warnings.warn(exceptions.CannotStoreOutputWarning(err))
        else:
            self.outputs_list.append(outputs)


class ModelRunnerWithObjective(ModelRunner[_Input, _Target, _Output]):
    """
    Run a model on a dataset and calculate the value of an objective function.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        objective: processes the model outputs and targets.
        outputs_list: list of optionally stored outputs.
    """

    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            name: str = '',
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            objective: p.ObjectiveProtocol[_Output, _Target],
    ) -> None:
        """
        Args:
            model: the model containing the weights to evaluate.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            loader: provides inputs and targets in batches.
            objective: processes the model outputs and targets.

        """
        super().__init__(model, loader=loader, name=name)
        self.objective = copy.deepcopy(objective)
        self.objective.reset()
        return

    @override
    def _get_metrics(self) -> Mapping[str, float]:
        return metrics.repr_metrics(self.objective)

    @override
    def _run_epoch(self, store_outputs: bool):
        self.objective.reset()  # reset before epoch to keep last metrics
        super()._run_epoch(store_outputs)
        return

    @override
    def _run_backward(self, outputs: _Output, targets: _Target) -> None:
        self.objective.update(outputs, targets)
        super()._run_backward(outputs, targets)
        return


class ModelRunnerWithLogs(ModelRunnerWithObjective[_Input, _Target, _Output],
                          Source):
    """
    Run a model on a dataset and log the value of an objective function.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        objective: processes the model outputs and targets.
        outputs_list: list of optionally stored outputs.
    """

    def _run_epoch(self, store_outputs: bool):
        super()._run_epoch(store_outputs)
        log_events.Metrics(model_name=self.model.name,
                           source_name=self.name,
                           epoch=self.model.epoch,
                           metrics=self._get_metrics())
        return

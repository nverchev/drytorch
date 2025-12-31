"""Module containing classes that run a model."""

import abc
import copy
import sys
import warnings

from collections.abc import Iterator, Mapping
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from torch import distributed as dist
from torch.utils import data
from typing_extensions import override

from drytorch.core import exceptions, log_events, register
from drytorch.core import protocols as p
from drytorch.lib import load, objectives
from drytorch.utils import apply_ops, repr_utils


__all__ = [
    'ModelCaller',
    'ModelRunner',
    'ModelRunnerWithLogs',
    'ModelRunnerWithObjective',
]


Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)
_Objective_co = TypeVar(
    '_Objective_co', bound=p.LossProtocol[Any, Any], covariant=True
)


@runtime_checkable
class SupportsSync(Protocol):
    """Protocol for objects that support syncing."""

    def sync(self) -> None:
        """Sync across all processes."""


class ModelCaller(
    repr_utils.CreatedAtMixin, Generic[Input, Output], metaclass=abc.ABCMeta
):
    """Base class that calls a model.

    Attributes:
        model: the wrapped model.
    """

    _name = repr_utils.DefaultName()

    model: p.ModelProtocol[Input, Output]

    def __init__(
        self, model: p.ModelProtocol[Input, Output], name: str = ''
    ) -> None:
        """Initialize.

        Args:
            model: the wrapped model.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
        """
        super().__init__()
        self.model: Final = model
        self._name = name
        return

    @property
    def name(self) -> str:
        """The name of the model."""
        return self._name

    @abc.abstractmethod
    def __call__(self) -> None:
        """Document itself when the model is first called."""
        register.register_actor(self, self.model)
        return

    @override
    def __repr__(self) -> str:
        return f'{self.name}({self.model.name})'


class ModelRunner(ModelCaller[Input, Output], Generic[Input, Target, Output]):
    """Run a model on a dataset.

    Attributes:
        model: the model to run.
        loader: the loader providing inputs and targets in batches.
        outputs_list: list of optionally stored outputs.
    """

    max_stored_output: ClassVar[int] = sys.maxsize

    loader: p.LoaderProtocol[tuple[Input, Target]]
    outputs_list: list[Output]
    _cached_metrics: Mapping[str, float]
    _is_distributed: bool
    _world_size: int
    _max_stored_process: int

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        name: str = '',
        *,
        loader: p.LoaderProtocol[tuple[Input, Target]],
    ) -> None:
        """Initialize.

        Args:
            model: the model to run.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            loader: provides inputs and targets in batches.

        """
        super().__init__(model, name)
        self.loader: Final = loader
        self.outputs_list: Final = list[Output]()
        self._cached_metrics = {}
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._is_distributed else 1
        self._max_stored_process = self.max_stored_output // self._world_size
        return

    def __call__(self, store_outputs: bool = False) -> None:
        """Single pass on the dataset.

        Args:
            store_outputs: whether to store model outputs. Defaults to False.
        """
        super().__call__()
        self._run_epoch(store_outputs)
        return

    @property
    def computed_metrics(self) -> Mapping[str, float]:
        """Retrieve cached metrics."""
        return self._cached_metrics

    def _compute_metrics(self) -> Mapping[str, float]:
        return {}

    def _get_batches(self) -> Iterator[tuple[Input, Target]]:
        if self._is_distributed:
            if isinstance(self.loader.sampler, data.DistributedSampler):
                self.loader.sampler.set_epoch(self.model.epoch)

        return (
            apply_ops.apply_to(batch, self.model.device)
            for batch in self.loader
        )

    def _run_backward(self, outputs: Output, targets: Target) -> None:
        _not_used = outputs, targets
        return

    def _run_batch(
        self,
        batch: tuple[Input, Target],
    ) -> Output:
        inputs, targets = batch
        outputs = self._run_forward(inputs)
        self._run_backward(outputs, targets)
        return outputs

    def _run_epoch(self, store_outputs: bool):
        if self._is_distributed:
            if not hasattr(self.model.module, 'module'):
                warnings.warn(
                    exceptions.ModuleNotDistributedWarning(), stacklevel=2
                )

        self.outputs_list.clear()
        n_samples = load.validate_dataset_length(self.loader.get_dataset())
        n_processes = dist.get_world_size() if self._is_distributed else 1
        pbar = log_events.IterateBatchEvent(
            self.name, self.loader.batch_size, len(self.loader), n_samples
        )
        for batch in self._get_batches():
            outputs = self._run_batch(batch)
            pbar.update(self._compute_metrics(), n_processes)
            if store_outputs:
                self._store(outputs)

        if self._is_distributed:
            self._sync()
            pbar.update(self._compute_metrics(), 0)  # trigger sync
            self._gather_stored_outputs()

        return

    def _run_forward(self, inputs: Input) -> Output:
        return self.model(inputs)

    def _store(self, outputs: Output) -> None:
        try:
            outputs = apply_ops.apply_cpu_detach(outputs)
        except (
            exceptions.FuncNotApplicableError,
            exceptions.NamedTupleOnlyError,
        ) as err:
            warnings.warn(
                exceptions.CannotStoreOutputWarning(err), stacklevel=3
            )
        else:
            if len(self.outputs_list) < self._max_stored_process:
                self.outputs_list.append(outputs)

        return

    def _sync(self) -> None:
        """Synchronize objective across processes."""
        return None

    def _gather_stored_outputs(self) -> None:
        """Gather outputs from all processes to rank 0."""
        if not self.outputs_list:
            return

        rank = dist.get_rank()
        try:
            if rank == 0:
                dist_outputs: list[list[Output]] = [[]] * self._world_size
                dist.gather_object(self.outputs_list, dist_outputs, dst=0)
                self.outputs_list.clear()
                for process_outputs in dist_outputs:
                    if process_outputs:
                        self.outputs_list.extend(process_outputs)
            else:
                dist.gather_object(self.outputs_list, None, dst=0)
                self.outputs_list.clear()

        except Exception as err:
            warnings.warn(
                exceptions.DistributedStorageWarning(err), stacklevel=2
            )

        return


class ModelRunnerWithObjective(
    ModelRunner[Input, Target, Output],
    p.MonitorProtocol,
    Generic[Input, Target, Output, _Objective_co],
):
    """Run a model on a dataset, calculating the value of an objective function.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        objective: processes the model outputs and targets.
        outputs_list: list of optionally stored outputs.
    """

    objective: _Objective_co

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        name: str = '',
        *,
        loader: p.LoaderProtocol[tuple[Input, Target]],
        objective: _Objective_co,
    ) -> None:
        """Initialize.

        Args:
            model: the model containing the weights to evaluate.
            name: the name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            loader: provides inputs and targets in batches.
            objective: processes the model outputs and targets.

        """
        super().__init__(model, loader=loader, name=name)
        self.objective: Final = copy.deepcopy(objective)
        if self._is_distributed:
            if getattr(self.objective, 'sync_on_compute', False):
                issue = 'sync_on_compute=True will cause overhead'
                recommend = 'set sync_on_compute to False'
                warnings.warn(
                    exceptions.ObjectiveSyncWarning(issue, recommend),
                    stacklevel=2,
                )
            if getattr(self.objective, 'dist_sync_on_step', False):
                issue = 'dist_sync_on_step=True will cause overhead'
                recommend = 'set dist_sync_on_step to False'
                warnings.warn(
                    exceptions.ObjectiveSyncWarning(issue, recommend),
                    stacklevel=2,
                )

        self.objective.reset()
        return

    def _compute_metrics(self) -> Mapping[str, float]:
        self._cached_metrics = objectives.compute_metrics(self.objective)
        return self._cached_metrics

    @override
    def _run_epoch(self, store_outputs: bool):
        self.objective.reset()  # reset before epoch to keep last metrics
        super()._run_epoch(store_outputs)
        return

    @override
    def _run_backward(self, outputs: Output, targets: Target) -> None:
        self.objective.update(outputs, targets)
        super()._run_backward(outputs, targets)
        return

    @override
    def _sync(self) -> None:
        if isinstance(self.objective, SupportsSync):
            self.objective.sync()
        else:
            issue = 'metrics not synchronized (averaged) across processes'
            recommend = "override Runner's `_sync` method"
            warnings.warn(
                exceptions.ObjectiveSyncWarning(issue, recommend),
                stacklevel=2,
            )


class ModelRunnerWithLogs(
    ModelRunnerWithObjective[Input, Target, Output, _Objective_co]
):
    """Run a model on a dataset and log the value of an objective function.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        objective: processes the model outputs and targets.
        outputs_list: list of optionally stored outputs.
    """

    def _run_epoch(self, store_outputs: bool):
        super()._run_epoch(store_outputs)
        log_events.MetricEvent(
            model_name=self.model.name,
            source_name=self.name,
            epoch=self.model.epoch,
            metrics=self._compute_metrics(),
        )
        return

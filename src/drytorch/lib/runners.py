"""Classes to run a model."""

import abc
import copy
import sys
import warnings

from collections.abc import Iterator, Mapping, Sequence
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch

from torch import distributed as dist
from torch.utils import data
from typing_extensions import override

from drytorch.core import exceptions, log_events, registering
from drytorch.core import protocols as p
from drytorch.lib import loading, objectives
from drytorch.utils import apply_ops, repr_utils


__all__ = [
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


class ModelRunner(
    repr_utils.CreatedAtMixin,
    Generic[Input, Target, Output],
    metaclass=registering.RegisteringMeta,
):
    """Run a model on a dataset.

    The runner registers as an actor of its model once created, so it must be
    created during the run that registered the model.

    Attributes:
        loader: the loader providing inputs and targets in batches.
        max_stored_output: maximum number of batch outputs stored in memory.
        model: the model to run.
    """

    max_stored_output: ClassVar[int] = sys.maxsize

    _name = repr_utils.DefaultName()

    loader: Final[p.LoaderProtocol[tuple[Input, Target]]]
    model: Final[p.ModelProtocol[Input, Output]]
    _cached_metrics: Mapping[str, float]
    _is_distributed: bool
    _max_stored_process: int
    _outputs_list: Final[list[Output]]
    _world_size: int

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
        super().__init__()
        self.loader = loader
        self.model = model
        self._name = name
        self._cached_metrics = {}
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._is_distributed else 1
        self._max_stored_process = self.max_stored_output // self._world_size
        self._outputs_list = []
        return

    def __call__(self, store_outputs: bool = False) -> None:
        """Single pass on the dataset.

        Args:
            store_outputs: whether to store model outputs. Defaults to
                ``False``.
        """
        self._run_epoch(store_outputs)
        return

    @property
    def computed_metrics(self) -> Mapping[str, float]:
        """Retrieve cached metrics."""
        return self._cached_metrics

    @property
    def name(self) -> str:
        """The name of the runner."""
        return self._name

    @property
    def outputs_list(self) -> Sequence[Output]:
        """Sequence of optionally stored outputs."""
        return self._outputs_list

    @override
    def __repr__(self) -> str:
        return f'{self.name}({self.model.name})'

    def _check_divisibility(self, n_samples: int) -> None:
        if torch.is_inference_mode_enabled() and self._is_distributed:
            if n_samples % self._world_size:
                warnings.warn(
                    exceptions.DistributedDatasetNotDivisibleWarning(
                        self.name, n_samples, self._world_size
                    ),
                    stacklevel=2,
                )

        return

    def _compute_metrics(self) -> Mapping[str, float]:
        return {}

    def _gather_stored_outputs(self) -> None:
        if not self._outputs_list:
            return

        rank = dist.get_rank()
        try:
            if rank == 0:
                dist_outputs: list[list[Output]] = [
                    [] for _ in range(self._world_size)
                ]
                dist.gather_object(self._outputs_list, dist_outputs, dst=0)
                self._outputs_list.clear()
                for gathered_tuple in zip(*dist_outputs, strict=True):
                    self._outputs_list.extend(gathered_tuple)
            else:
                dist.gather_object(self._outputs_list, None, dst=0)
                self._outputs_list.clear()

        except Exception as err:
            warnings.warn(
                exceptions.DistributedStorageWarning(err), stacklevel=2
            )

        return

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
            if isinstance(self.model, _ExecModuleProtocol):
                exec_module = self.model.exec_module
            else:
                exec_module = None

            if exec_module is not None and not isinstance(
                exec_module, torch.nn.parallel.DistributedDataParallel
            ):
                warnings.warn(
                    exceptions.ModuleNotDistributedWarning(), stacklevel=2
                )

        self._outputs_list.clear()
        n_samples = loading.validate_dataset_length(self.loader.dataset)
        self._check_divisibility(n_samples)
        n_processes = dist.get_world_size() if self._is_distributed else 1
        batch_event = log_events.IterateBatchEvent(
            self.name, self.loader.batch_size, len(self.loader), n_samples
        )
        for batch in self._get_batches():
            outputs = self._run_batch(batch)
            metrics = self._compute_metrics()
            batch_event.update(metrics, n_processes)
            if store_outputs:
                self._store(outputs)

        if self._is_distributed:
            self._sync()
            batch_event.refresh(self._compute_metrics())
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
            if len(self._outputs_list) < self._max_stored_process:
                self._outputs_list.append(outputs)

        return

    def _sync(self) -> None:
        return None


class ModelRunnerWithObjective(
    ModelRunner[Input, Target, Output],
    Generic[Input, Target, Output, _Objective_co],
):
    """Run a model on a dataset, calculating the value of an objective function.

    Attributes:
        loader: provides inputs and targets in batches.
        model: the model containing the weights to evaluate.
        objective: processes the model outputs and targets.
    """

    objective: Final[_Objective_co]
    _checked_metric_device: bool

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
        self.objective = copy.deepcopy(objective)
        if self._is_distributed:
            if isinstance(self.objective, _SyncOnComputeProtocol):
                if self.objective.sync_on_compute:
                    issue = 'sync_on_compute=True will cause overhead'
                    recommend = 'set sync_on_compute to False'
                    warnings.warn(
                        exceptions.ObjectiveSyncWarning(issue, recommend),
                        stacklevel=2,
                    )

            if isinstance(self.objective, _DistSyncOnStepProtocol):
                if self.objective.dist_sync_on_step:
                    issue = 'dist_sync_on_step=True will cause overhead'
                    recommend = 'set dist_sync_on_step to False'
                    warnings.warn(
                        exceptions.ObjectiveSyncWarning(issue, recommend),
                        stacklevel=2,
                    )

        self.objective.reset()
        self._checked_metric_device = False
        return

    def _compute_metrics(self) -> Mapping[str, float]:
        if self._is_distributed and not self._checked_metric_device:
            try:
                objectives.check_device(self.objective, self.model.device)
            except exceptions.DeviceMismatchError as err:
                raise exceptions.ModelDeviceMismatchError() from err

            self._checked_metric_device = True

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
        if isinstance(self.objective, p.SyncProtocol):
            self.objective.sync()
        else:
            issue = 'metrics not synchronized (averaged) across processes'
            recommend = "override Runner's `_sync` method"
            warnings.warn(
                exceptions.ObjectiveSyncWarning(issue, recommend),
                stacklevel=2,
            )

        return


class ModelRunnerWithLogs(
    ModelRunnerWithObjective[Input, Target, Output, _Objective_co]
):
    """Run a model on a dataset and log the value of an objective function.

    Attributes:
        loader: provides inputs and targets in batches.
        model: the model containing the weights to evaluate.
        objective: processes the model outputs and targets.
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


# ---------------------------------------------------------------------------
# Private protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class _DistSyncOnStepProtocol(Protocol):
    """Protocol for objectives exposing a dist_sync_on_step flag."""

    @property
    @abc.abstractmethod
    def dist_sync_on_step(self) -> bool:
        """Flag whether to sync on step."""


@runtime_checkable
class _ExecModuleProtocol(Protocol):
    """Protocol for models exposing an execution module."""

    @property
    @abc.abstractmethod
    def exec_module(self) -> torch.nn.Module:
        """The executed module."""


@runtime_checkable
class _SyncOnComputeProtocol(Protocol):
    """Protocol for objectives exposing a sync_on_compute flag."""

    @property
    @abc.abstractmethod
    def sync_on_compute(self) -> bool:
        """Flag whether to sync on compute."""

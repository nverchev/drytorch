"""Module containing registry, callbacks, and hooks for a Trainer."""

from __future__ import annotations

import abc
import functools
import operator

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Generic, Literal, ParamSpec, TypeVar, cast

from typing_extensions import override

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import objectives, schedulers
from drytorch.lib.schedulers import RestartScheduler


__all__ = [
    'CallEvery',
    'ChangeSchedulerOnPlateauCallback',
    'EarlyStoppingCallback',
    'Hook',
    'HookRegistry',
    'MetricExtractor',
    'MetricMonitor',
    'PruneCallback',
    'ReduceLROnPlateau',
    'RestartScheduleOnPlateau',
    'StaticHook',
    'TrainerHook',
    'call_every',
    'saving_hook',
    'static_hook_class',
]

_T_contra = TypeVar('_T_contra', contravariant=True)
_P = ParamSpec('_P')
_Q = ParamSpec('_Q')
Input = TypeVar('Input', bound=p.InputType, contravariant=True)
Target = TypeVar('Target', bound=p.TargetType, contravariant=True)
Output = TypeVar('Output', bound=p.OutputType, contravariant=True)
get_last: Final = operator.itemgetter(-1)


class HookRegistry(Generic[_T_contra]):
    """A registry for managing and executing hooks.

    The hooks have a generic object as input and can access it.

    Attributes:
        hooks: a list of registered hooks.
    """

    hooks: list[Callable[[_T_contra], None]]

    def __init__(self) -> None:
        """Initialize."""
        self.hooks = []

    def execute(self, input_object: _T_contra) -> None:
        """Execute the registered hooks in order of registration.

        Args:
            input_object: the input to pass to each hook.
        """
        for hook in self.hooks:
            hook(input_object)
        return

    def register(self, hook: Callable[[_T_contra], None]) -> None:
        """Register a single hook.

        Args:
            hook: the hook to register.
        """
        self.hooks.append(hook)
        return

    def register_all(
        self,
        hook_list: list[Callable[[_T_contra], None]],
    ) -> None:
        """Register multiple hooks.

        Args:
            hook_list: the list of hooks to register.
        """
        for hook in hook_list:
            self.register(hook)
        return


class TrainerHook(
    Generic[Input, Target, Output],
    metaclass=abc.ABCMeta,
):
    """Compose control flow on side effects while keeping track of parameters.

    Args:
        parameters: metadata associated with the hook
        base_hook_name: name of the base hook for representation
    """

    parameters: dict[str, Any]
    base_hook_name: str

    @abc.abstractmethod
    def __call__(
        self,
        trainer: p.TrainerProtocol[Input, Target, Output],
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """

    def bind(
        self,
        f: Callable[
            [Callable[[p.TrainerProtocol[Input, Target, Output]], None]],
            Hook,
        ],
        /,
    ) -> Hook[Input, Target, Output]:
        """Allow transformation of the Hook.

        Args:
            f: a function specifying the transformation.

        Returns:
            the transformed Hook.
        """
        next_hook = f(self.__call__)
        parameters = self.parameters | next_hook.parameters
        return Hook(
            wrapped=next_hook.__call__,
            parameters=parameters,
            base_hook_name=self.base_hook_name,
        )


class Hook(TrainerHook[Input, Target, Output]):
    """Wrapper for callable taking a Trainer as input.

    Attributes:
        parameters: metadata associated with the hook.
        base_hook_name: name of the base hook for representation.
    """

    def __init__(
        self,
        wrapped: Callable[[p.TrainerProtocol[Input, Target, Output]], None],
        base_hook_name: str | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize.

        Args:
            wrapped: the function to be conditionally called.
            base_hook_name: name of the base hook for representation.
                Defaults to the class name.
            parameters: metadata associated with the hook.
        """
        self._wrapped = wrapped
        self.parameters = parameters or {}
        self.base_hook_name = base_hook_name or self.__class__.__name__
        return

    def __call__(
        self,
        trainer: p.TrainerProtocol[Input, Target, Output],
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
        self._wrapped(trainer)
        return


class StaticHook(TrainerHook[Any, Any, Any]):
    """Ignoring arguments and execute a wrapped function.

    Attributes:
        parameters: metadata associated with the hook.
        base_hook_name: name of the base hook for representation.
    """

    def __init__(
        self,
        wrapped: Callable[[], None],
        base_hook_name: str | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """Initialize.

        Args:
            wrapped: the function to be wrapped and called statically.
            base_hook_name: name of the base hook for representation.
                Defaults to the class name.
            parameters: metadata associated with the hook.
        """
        self._wrapped = wrapped
        self.parameters = parameters or {}
        self.base_hook_name = base_hook_name or self.__class__.__name__
        return

    def __call__(
        self,
        trainer: p.TrainerProtocol[Input, Target, Output],
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
        self._wrapped()
        return


class CallEvery(Hook[Input, Target, Output]):
    """Metadata-aware wrapper for periodic execution.

    Attributes:
        parameters: metadata associated with the hook.
    """

    def __init__(
        self,
        wrapped: Callable[[p.TrainerProtocol[Input, Target, Output]], None],
        parameters: dict[str, Any] | None = None,
        interval: int = 1,
        start: int = 0,
    ):
        """Initialize.

        Args:
            wrapped: the function to be wrapped and called statically.
            parameters: metadata associated with the hook.
            interval: the frequency of calling the function.
            start: the epoch to start calling the function.
        """
        super().__init__(wrapped, '', parameters)
        if interval <= 0:
            raise ValueError(f'interval must be positive, got {interval}')

        if start < 0:
            raise ValueError(f'start must be non-negative, got {start}')

        self.start = start
        self.interval = interval
        self.parameters['interval'] = interval
        self.parameters['start'] = start
        return

    def __call__(
        self,
        trainer: p.TrainerProtocol[Input, Target, Output],
    ) -> None:
        """Optionally, calling wrapped callable."""
        epoch = trainer.model.epoch
        if epoch < self.start:
            return

        if not (epoch - self.start) % self.interval or trainer.terminated:
            self._wrapped(trainer)

        return


def call_every(
    interval: int,
    start: int = 0,
) -> Callable[
    [Callable[[p.TrainerProtocol[Input, Target, Output]], None]],
    Hook[Input, Target, Output],
]:
    """Create a transformer for periodic hook execution.

    Args:
        interval: the frequency of calling the hook.
        start: the epoch to start calling the hook.

    Returns:
        A transformer that adds periodic logic to a callable.
    """

    def _transformer(
        func: Callable[[p.TrainerProtocol[Input, Target, Output]], None],
    ) -> CallEvery[Input, Target, Output]:
        return CallEvery(wrapped=func, interval=interval, start=start)

    return _transformer


def static_hook_class(
    cls: Callable[_P, Callable[[], None]],
) -> Callable[_P, StaticHook]:
    """Class decorator to wrap a callable class into a static hook type.

    Args:
        cls: a callable class that takes no arguments and returns None.

    Returns:
        A class that can be instantiated in the same way to have a static hook.
    """
    if not isinstance(cls, type):
        raise TypeError(f'Input must be a class, got {type(cls)}')

    @functools.wraps(cls, updated=())
    class _StaticHookDecorator(StaticHook):
        @override
        def __init__(self, *args: _P.args, **kwargs: _P.kwargs) -> None:
            self._init_args: tuple[Any, ...] = args
            self._init_kwargs: dict[str, Any] = dict(kwargs)
            metadata: dict[str, Any] = dict(kwargs)
            if args:
                metadata['args'] = args

            super().__init__(cls(*args, **kwargs), cls.__name__, metadata)
            return

        def __reduce__(
            self,
        ) -> tuple[
            Callable[[type, tuple[Any, ...], dict[str, Any]], StaticHook],
            tuple[type, tuple[Any, ...], dict[str, Any]],
        ]:
            bricks = (cast(type, cls), self._init_args, self._init_kwargs)
            return _unpickle_static_hook, bricks

    return _StaticHookDecorator


@Hook
def saving_hook(trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
    """Create a hook that saves the model's checkpoint.

    Args:
        trainer: the trainer instance.
    """
    trainer.save_checkpoint()
    return


class MetricExtractor:
    """Handle extraction of metrics from trainer/validation protocols.

    This class is responsible for interfacing with trainer and validation
    protocols to extract metric values.

    Attributes:
        metric_spec: the metric specification (name or protocol instance).
        optional_monitor: evaluation protocol to monitor.
    """

    metric_spec: p.ObjectiveProtocol[Any, Any] | str | None
    optional_monitor: p.MonitorProtocol | None
    _resolved_metric_name: str | None

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Any, Any] | str | None = None,
        monitor: p.MonitorProtocol | None = None,
    ) -> None:
        """Initialize.

        Args:
            metric: name of the metric to monitor or metric calculator instance.
            monitor: evaluation protocol to monitor.
        """
        self.metric_spec: Final = metric
        self.optional_monitor = monitor
        self._resolved_metric_name = None

    @property
    def metric_name(self) -> str | None:
        """Get the resolved metric name."""
        return self._resolved_metric_name

    def extract_metric_value(
        self,
        instance: p.TrainerProtocol[Input, Target, Output],
        tracker: objectives.MetricTracker[Output, Target],
    ) -> float:
        """Extract and return the metric value from the instance.

        Args:
            instance: Trainer instance to extract from.
            tracker: objectives.MetricTracker to potentially update metric name.

        Returns:
            The extracted metric value.

        Raises:
            MetricNotFoundError: if the specified metric is not found.
        """
        monitor = self._get_monitor(instance)
        last_metrics = monitor.computed_metrics

        if self._resolved_metric_name is None:
            if self.metric_spec is None:
                self._resolved_metric_name = next(iter(last_metrics.keys()))
            else:
                self._resolved_metric_name = self._get_metric_name(
                    self.metric_spec
                )

            tracker.metric_name = self._resolved_metric_name

        if self._resolved_metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(
                monitor.name, self._resolved_metric_name
            )

        return last_metrics[self._resolved_metric_name]

    def get_metric_best_is(self) -> Literal['auto', 'higher', 'lower'] | None:
        """Get the best_is preference from the metric if available."""
        return self._get_metric_best_is(self.metric_spec)

    def _get_monitor(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> p.MonitorProtocol:
        if self.optional_monitor is None:
            if instance.validation is None:
                return instance

            return instance.validation

        return self.optional_monitor

    @staticmethod
    def _get_metric_name(
        metric: p.ObjectiveProtocol[Output, Target] | str,
    ) -> str:
        if isinstance(metric, str):
            return metric

        if name := getattr(metric, 'name', False):
            return str(name)

        if name := getattr(metric, '_get_name', False):
            return str(name)

        return metric.__class__.__name__

    @staticmethod
    def _get_metric_best_is(
        metric: p.ObjectiveProtocol[Output, Target] | str | None,
    ) -> Literal['auto', 'higher', 'lower'] | None:
        higher_is_better = getattr(metric, 'higher_is_better', None)
        if higher_is_better is None:
            return None

        return 'higher' if higher_is_better else 'lower'


class MetricMonitor(Generic[Output, Target]):
    """Handle metric monitoring and alerts when performance stops increasing.

    Attributes:
        metric_tracker: handles metric value tracking and improvement detection.
        extractor: handles metric extraction from protocols.
    """

    extractor: MetricExtractor
    metric_tracker: objectives.MetricTracker[Output, Target]

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """Initialize.

        Args:
            metric: name of the metric to monitor or metric calculator instance.
            monitor: evaluation protocol to monitor.
            min_delta: minimum change required to qualify as an improvement.
            patience: number of checks to wait before triggering callback.
            best_is: whether higher or lower metric values are better.
            filter_fn: function to aggregate recent metric values.
        """
        self.extractor: Final = MetricExtractor(metric=metric, monitor=monitor)

        metric_best_is = self.extractor.get_metric_best_is()
        if metric_best_is is not None:
            best_is = metric_best_is

        initial_metric_name = None
        if isinstance(metric, str):
            initial_metric_name = metric

        self.metric_tracker = objectives.MetricTracker(
            metric_name=initial_metric_name,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
        )

    @property
    def metric_name(self) -> str | None:
        """Get the metric name being monitored."""
        return self.extractor.metric_name or self.metric_tracker.metric_name

    @property
    def best_value(self) -> float:
        """Get the best result observed so far."""
        return self.metric_tracker.best_value

    @property
    def filtered_value(self) -> float:
        """Get the current filtered value."""
        return self.metric_tracker.filtered_value

    @property
    def history(self) -> list[float]:
        """Get the metric history."""
        return self.metric_tracker.history

    def is_better(self, value: float, reference: float) -> bool:
        """Check whether to be patient."""
        return self.metric_tracker.is_better(value, reference)

    def is_improving(self) -> bool:
        """Determine if the model performance is improving."""
        return self.metric_tracker.is_improving()

    def is_patient(self) -> bool:
        """Check whether to be patient."""
        return self.metric_tracker.is_patient()

    def record_metric_value(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> None:
        """Register a new metric value from a monitored evaluation.

        Args:
            instance: Trainer instance to extract metric from.

        Raises:
            MetricNotFoundError: if the specified metric is not found.
        """
        value = self.extractor.extract_metric_value(
            instance, self.metric_tracker
        )
        self.metric_tracker.add_value(value)


class EarlyStoppingCallback(Generic[Output, Target]):
    """Implement early stopping logic for training models.

    Attributes:
        monitor: monitor instance.
        start_from_epoch: epoch to start monitoring from.
    """

    monitor: MetricMonitor[Output, Target]
    start_from_epoch: int

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 10,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
        start_from_epoch: int = 2,
    ) -> None:
        """Initialize.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            monitor: evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: minimum change required to qualify as an improvement.
            patience: number of calls to wait before stopping.
                Default 'auto' will determine this from initial measurements.
            best_is: whether higher or lower metric values are better.
            filter_fn: function to aggregate recent metric values. Default
                gets the last value.
            start_from_epoch: first epoch to start monitoring from.
        """
        self.monitor: Final = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.start_from_epoch: Final = start_from_epoch
        return

    def __call__(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> None:
        """Evaluate whether training should be stopped early.

        Args:
            instance: Trainer instance to evaluate.
        """
        self.monitor.record_metric_value(instance)
        epoch = instance.model.epoch
        if epoch < self.start_from_epoch:
            return

        if self.monitor.is_improving() or self.monitor.is_patient():
            return

        best_result = self.monitor.best_value
        metric_name = self.monitor.metric_name
        msg = f'Training stopped with best result={best_result} {metric_name}'
        instance.terminate_training(msg)
        return


class PruneCallback(Generic[Output, Target]):
    """Implement pruning logic for training models.

    Attributes:
        monitor: monitor instance.
        thresholds: dictionary mapping epochs to pruning thresholds.
    """

    monitor: MetricMonitor[Output, Target]
    thresholds: Mapping[int, float | None]
    trial_values: dict[int, float]

    def __init__(
        self,
        thresholds: Mapping[int, float | None],
        metric: str | p.ObjectiveProtocol[Output, Target] | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """Initialize.

        Args:
            thresholds: dictionary mapping epochs to pruning values.
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            monitor: evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: minimum change required to qualify as an improvement.
            best_is: whether higher or lower metric values are better.
               Default 'auto' will determine this from initial measurements.
            filter_fn: function to aggregate the intermediate results
            values. Default
                gets the last value.
        """
        self.monitor: Final = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.thresholds: Final = thresholds
        self.trial_values = {}
        return

    def __call__(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> None:
        """Evaluate whether training should be stopped early.

        Args:
            instance: trainer instance to evaluate.
        """
        self.monitor.record_metric_value(instance)
        epoch = instance.model.epoch
        if epoch not in self.thresholds:
            return

        threshold = self.thresholds[epoch]
        value = self.monitor.filtered_value
        if threshold is None or not self.monitor.is_better(value, threshold):
            self.trial_values[epoch] = value
            metric_name = self.monitor.metric_name
            msg = f'Training stopped at {threshold=} {metric_name}'
            instance.terminate_training(msg)

        return


class ChangeSchedulerOnPlateauCallback(
    Generic[Output, Target], metaclass=abc.ABCMeta
):
    """Change the learning rate schedule when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
    """

    monitor: MetricMonitor[Output, Target]
    cooldown: int
    _cooldown_counter: int

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
        cooldown: int = 0,
    ) -> None:
        """Initialize.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            monitor: evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: minimum change required to qualify as an improvement.
            patience: number of checks to wait before changing the schedule.
            best_is: whether higher or lower metric values are better.
                Default 'auto' will determine this from initial measurements.
            filter_fn: function to aggregate recent metric values. Default
                gets the last value.
            cooldown: calls to skip after changing the schedule.
        """
        self.monitor: Final = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.cooldown: Final = cooldown
        self._cooldown_counter = 0
        return

    def __call__(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> None:
        """Check if there is a plateau and reduce the learning rate if needed.

        Args:
            instance: Trainer instance to evaluate.
        """
        self.monitor.record_metric_value(instance)
        epoch = instance.model.epoch

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        if self.monitor.is_improving() or self.monitor.is_patient():
            return

        scheduler = self.get_scheduler(
            epoch, instance.learning_schema.scheduler
        )
        instance.update_learning_rate(base_lr=None, scheduler=scheduler)
        self._cooldown_counter = self.cooldown  # start the cooldown period
        return

    @abc.abstractmethod
    def get_scheduler(
        self, epoch: int, scheduler: p.SchedulerProtocol
    ) -> p.SchedulerProtocol:
        """Modify input scheduler.

        Args:
            epoch: current epoch.
            scheduler: scheduler to be modified.

        Returns:
            Modified scheduler.
        """


class ReduceLROnPlateau(ChangeSchedulerOnPlateauCallback[Output, Target]):
    """Reduce the learning rate when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
        factor: factor by which to reduce the learning rate.
    """

    factor: float

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
        factor: float = 0.1,
        cooldown: int = 0,
    ) -> None:
        """Initialize.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            monitor: evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: minimum change required to qualify as an improvement.
            patience: number of checks to wait before changing the schedule.
            best_is: whether higher or lower metric values are better.
                Default 'auto' will determine this from initial measurements.
            filter_fn: function to aggregate recent metric values. Default
                gets the last value.
            cooldown: calls to skip after changing the schedule.
            factor: factor by which to reduce the learning rate.
        """
        super().__init__(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
            cooldown=cooldown,
        )
        self.factor: Final = factor

    def get_scheduler(
        self, epoch: int, scheduler: p.SchedulerProtocol
    ) -> p.SchedulerProtocol:
        """Modify the input scheduler to scale down the learning rate.

        Args:
            epoch: not used.
            scheduler: scheduler to be modified.

        Returns:
            Modified scheduler.
        """
        return schedulers.RescaleScheduler(scheduler, self.factor)


class RestartScheduleOnPlateau(
    ChangeSchedulerOnPlateauCallback[Output, Target]
):
    """Restart the scheduling after plateauing.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
    """

    def get_scheduler(
        self, epoch: int, scheduler: p.SchedulerProtocol
    ) -> p.SchedulerProtocol:
        """Consider training until now a warm-up and restart scheduling.

        Args:
            epoch: current epoch.
            scheduler: scheduler to be modified.

        Returns:
            Modified scheduler.
        """
        return RestartScheduler(scheduler, epoch)


def _unpickle_static_hook(
    cls: type,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> StaticHook:
    return static_hook_class(cls)(*args, **kwargs)

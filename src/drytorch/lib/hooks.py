"""Module containing registry, callbacks, and hooks for a Trainer."""

from __future__ import annotations

import abc
import operator

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Generic, Literal, ParamSpec, TypeVar

from typing_extensions import override

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import objectives, schedulers


_T_contra = TypeVar('_T_contra', contravariant=True)
_P = ParamSpec('_P')
_Q = ParamSpec('_Q')
_Input_contra = TypeVar('_Input_contra', bound=p.InputType, contravariant=True)
_Target_contra = TypeVar(
    '_Target_contra', bound=p.TargetType, contravariant=True
)
_Output_contra = TypeVar(
    '_Output_contra', bound=p.OutputType, contravariant=True
)
get_last: Final = operator.itemgetter(-1)


class HookRegistry(Generic[_T_contra]):
    """A registry for managing and executing hooks.

    The hooks have a generic object as input and can access it.

    Attributes:
        hooks: a list of registered hooks.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.hooks: list[Callable[[_T_contra], None]] = []

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
    Generic[_Input_contra, _Target_contra, _Output_contra],
    metaclass=abc.ABCMeta,
):
    """Callable supporting bind operations."""

    @abc.abstractmethod
    def __call__(
        self,
        trainer: p.TrainerProtocol[
            _Input_contra, _Target_contra, _Output_contra
        ],
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """

    def bind(
        self,
        f: Callable[
            [TrainerHook[_Input_contra, _Target_contra, _Output_contra]],
            TrainerHook[_Input_contra, _Target_contra, _Output_contra],
        ],
        /,
    ) -> TrainerHook[_Input_contra, _Target_contra, _Output_contra]:
        """Allow transformation of the Hook.

        Args:
            f: a function specifying the transformation.

        Returns:
            the transformed Hook.
        """
        return f(self)


class Hook(TrainerHook[_Input_contra, _Target_contra, _Output_contra]):
    """Wrapper for callable taking a Trainer as input."""

    def __init__(
        self,
        wrapped: Callable[
            [p.TrainerProtocol[_Input_contra, _Target_contra, _Output_contra]],
            None,
        ],
    ) -> None:
        """Constructor.

        Args:
            wrapped: the function to be conditionally called.
        """
        self.wrapped: Final = wrapped

    def __call__(
        self,
        trainer: p.TrainerProtocol[
            _Input_contra, _Target_contra, _Output_contra
        ],
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
        self.wrapped(trainer)


class StaticHook(TrainerHook[Any, Any, Any]):
    """Ignoring arguments and execute a wrapped function."""

    def __init__(self, wrapped: Callable[[], None]):
        """Constructor.

        Args:
            wrapped: the function to be wrapped and called statically.
        """
        self.wrapped: Callable[[], None] = wrapped

    def __call__(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Execute the call.

        Args:
            trainer: not used.
        """
        return self.wrapped()


class OptionalCallable(
    Hook[_Input_contra, _Target_contra, _Output_contra], metaclass=abc.ABCMeta
):
    """Abstract class for callables that execute based on custom conditions."""

    def __call__(
        self,
        trainer: p.TrainerProtocol[
            _Input_contra, _Target_contra, _Output_contra
        ],
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
        if self._should_call(trainer):
            return self.wrapped(trainer)

        return None

    @abc.abstractmethod
    def _should_call(
        self,
        trainer: p.TrainerProtocol[
            _Input_contra, _Target_contra, _Output_contra
        ],
    ) -> bool:
        """Determine if the callable should be executed."""


class CallEvery(
    OptionalCallable[_Input_contra, _Target_contra, _Output_contra]
):
    """Call a function at specified intervals."""

    def __init__(
        self,
        wrapped: Callable[
            [p.TrainerProtocol[_Input_contra, _Target_contra, _Output_contra]],
            None,
        ],
        interval: int,
        start: int,
    ) -> None:
        """Constructor.

        Args:
            start: the epoch to start calling the hook.
            interval: the frequency of calling the hook.
            wrapped: the function to be called periodically.
        """
        self.start: int = start
        self.interval: int = interval
        super().__init__(wrapped)
        return

    @override
    def _should_call(
        self,
        trainer: p.TrainerProtocol[
            _Input_contra, _Target_contra, _Output_contra
        ],
    ) -> bool:
        """Determine if the hook should be called based on the epoch.

        Args:
            trainer: the trainer instance containing epoch information.
        """
        epoch = trainer.model.epoch
        if epoch < self.start:
            return False

        return not (epoch - self.start) % self.interval or trainer.terminated


def call_every(
    interval: int,
    start: int = 0,
) -> Callable[
    [
        Callable[
            [p.TrainerProtocol[_Input_contra, _Target_contra, _Output_contra]],
            None,
        ]
    ],
    CallEvery[_Input_contra, _Target_contra, _Output_contra],
]:
    """Create a decorator for periodic hook execution.

    Args:
        start: the epoch to start calling the hook.
        interval: the frequency of calling the hook.

    Returns:
        A decorator that wraps a function in a CallEvery hook.
    """

    def _decorator(
        func: Callable[
            [p.TrainerProtocol[_Input_contra, _Target_contra, _Output_contra]],
            None,
        ],
    ) -> CallEvery[_Input_contra, _Target_contra, _Output_contra]:
        return CallEvery(func, interval, start)

    return _decorator


@Hook
def saving_hook(trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
    """Create a hook that saves the model's checkpoint.

    Args:
        trainer: the trainer instance.
    """
    trainer.save_checkpoint()
    return


def static_hook_class(
    cls: Callable[_P, Callable[[], None]],
) -> Callable[_P, StaticHook]:
    """Class decorator to wrap a callable class into a static hook type.

    Args:
        cls: a callable class that takes no arguments and returns None.

    Returns:
        A class that can be instantiated in the same way to have a static hook.
    """

    class _StaticHookDecorator(StaticHook):
        @override
        def __init__(self, *args: _P.args, **kwargs: _P.kwargs):
            super().__init__(cls(*args, **kwargs))

    return _StaticHookDecorator


class MetricExtractor:
    """Handle extraction of metrics from trainer/validation protocols.

    This class is responsible for interfacing with trainer and validation
    protocols to extract metric values.

    Attributes:
        metric_spec: the metric specification (name or protocol instance).
        optional_monitor: evaluation protocol to monitor.
    """

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Any, Any] | str | None = None,
        monitor: p.MonitorProtocol | None = None,
    ) -> None:
        """Constructor.

        Args:
            metric: name of the metric to monitor or metric calculator instance.
            monitor: evaluation protocol to monitor.
        """
        self.metric_spec: Final = metric
        self.optional_monitor: p.MonitorProtocol | None = monitor
        self._resolved_metric_name: str | None = None

    @property
    def metric_name(self) -> str | None:
        """Get the resolved metric name."""
        return self._resolved_metric_name

    def extract_metric_value(
        self,
        instance: p.TrainerProtocol[
            _Input_contra, _Target_contra, _Output_contra
        ],
        tracker: objectives.MetricTracker[_Output_contra, _Target_contra],
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
        self, instance: p.TrainerProtocol[Any, _Target_contra, _Output_contra]
    ) -> p.MonitorProtocol:
        if self.optional_monitor is None:
            if instance.validation is None:
                return instance
            return instance.validation
        return self.optional_monitor

    @staticmethod
    def _get_metric_name(
        metric: p.ObjectiveProtocol[_Output_contra, _Target_contra] | str,
    ) -> str:
        if isinstance(metric, str):
            return metric
        elif name := getattr(metric, 'name', False):
            return str(name)
        elif name := getattr(metric, '_get_name', False):
            return str(name)
        else:
            return metric.__class__.__name__

    @staticmethod
    def _get_metric_best_is(
        metric: p.ObjectiveProtocol[_Output_contra, _Target_contra]
        | str
        | None,
    ) -> Literal['auto', 'higher', 'lower'] | None:
        higher_is_better = getattr(metric, 'higher_is_better', None)
        if higher_is_better is None:
            return None
        else:
            return 'higher' if higher_is_better else 'lower'


class MetricMonitor(Generic[_Output_contra, _Target_contra]):
    """Handle metric monitoring and alerts when performance stops increasing.

    Attributes:
        metric_tracker: handles metric value tracking and improvement detection.
        extractor: handles metric extraction from protocols.
    """

    def __init__(
        self,
        metric: p.ObjectiveProtocol[_Output_contra, _Target_contra]
        | str
        | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """Constructor.

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

        self.metric_tracker: objectives.MetricTracker[
            _Output_contra, _Target_contra
        ] = objectives.MetricTracker(
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
        self, instance: p.TrainerProtocol[Any, _Target_contra, _Output_contra]
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


class EarlyStoppingCallback(Generic[_Output_contra, _Target_contra]):
    """Implement early stopping logic for training models.

    Attributes:
        monitor: monitor instance.
        start_from_epoch: start from epoch.
    """

    def __init__(
        self,
        metric: p.ObjectiveProtocol[_Output_contra, _Target_contra]
        | str
        | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 10,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
        start_from_epoch: int = 2,
    ) -> None:
        """Constructor.

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
        self.start_from_epoch = start_from_epoch
        return

    def __call__(
        self, instance: p.TrainerProtocol[Any, _Target_contra, _Output_contra]
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
        msg = f'Training stopped with best result={best_result} {metric_name}.'
        instance.terminate_training(msg)
        return


class PruneCallback(Generic[_Output_contra, _Target_contra]):
    """Implement pruning logic for training models.

    Attributes:
        monitor: monitor instance.
        thresholds: dictionary mapping epochs to pruning thresholds.
    """

    def __init__(
        self,
        thresholds: Mapping[int, float | None],
        metric: str
        | p.ObjectiveProtocol[_Output_contra, _Target_contra]
        | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """Constructor.

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
        self.thresholds = thresholds
        self.trial_values: dict[int, float] = {}
        return

    def __call__(
        self, instance: p.TrainerProtocol[Any, _Target_contra, _Output_contra]
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
            msg = f'Training stopped at {threshold=} {metric_name}.'
            instance.terminate_training(msg)

        return


class ChangeSchedulerOnPlateauCallback(
    Generic[_Output_contra, _Target_contra], metaclass=abc.ABCMeta
):
    """Change the learning rate schedule when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
    """

    def __init__(
        self,
        metric: p.ObjectiveProtocol[_Output_contra, _Target_contra]
        | str
        | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
        cooldown: int = 0,
    ) -> None:
        """Constructor.

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
        self.cooldown = cooldown
        self._cooldown_counter = 0
        return

    def __call__(
        self, instance: p.TrainerProtocol[Any, _Target_contra, _Output_contra]
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


class ReduceLROnPlateau(
    ChangeSchedulerOnPlateauCallback[_Output_contra, _Target_contra]
):
    """Reduce the learning rate when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
        factor: factor by which to reduce the learning rate.
    """

    def __init__(
        self,
        metric: p.ObjectiveProtocol[_Output_contra, _Target_contra]
        | str
        | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
        factor: float = 0.1,
        cooldown: int = 0,
    ) -> None:
        """Constructor.

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
        self.factor = factor

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
    ChangeSchedulerOnPlateauCallback[_Output_contra, _Target_contra]
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
        return schedulers.WarmupScheduler(scheduler, epoch)

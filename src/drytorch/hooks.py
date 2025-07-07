"""Module containing registry, callbacks, and hooks for a Trainer."""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import operator
from typing import Generic, Literal, Optional, ParamSpec, TypeVar, cast

from typing_extensions import override

from drytorch import exceptions
from drytorch import metrics
from drytorch import protocols as p
from drytorch import schedulers

_T = TypeVar('_T', contravariant=True)
_P = ParamSpec('_P')
_Q = ParamSpec('_Q')

get_last = operator.itemgetter(-1)


class HookRegistry(Generic[_T]):
    """
    A registry for managing and executing hooks.

    The hooks have a generic object as input and can access it.

    Attributes:
        hooks: a list of registered hooks.
    """

    def __init__(self) -> None:
        """
        Initialize the HookRegistry with an empty list of hooks.
        """
        self.hooks: list[Callable[[_T], None]] = []

    def execute(self, input_object: _T) -> None:
        """
        Execute the registered hooks in order of registration.

        Args:
            input_object: the input to pass to each hook.
        """
        for hook in self.hooks:
            hook(input_object)
        return

    def register(self, hook: Callable[[_T], None]) -> None:
        """
        Register a single hook.

        Args:
            hook: the hook to register.
        """
        self.hooks.append(hook)
        return

    def register_all(self, hook_list: list[Callable[[_T], None]]) -> None:
        """
        Register multiple hooks.

        Args:
            hook_list: the list of hooks to register.
        """
        for hook in hook_list:
            self.register(hook)
        return


class AbstractHook(metaclass=abc.ABCMeta):
    """Callable supporting bind operations."""

    @abc.abstractmethod
    def __call__(self, trainer: p.TrainerProtocol) -> None:
        """
        Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """

    def bind(self,
             f: Callable[[AbstractHook], AbstractHook],
             /,
             ) -> AbstractHook:
        """
        Allow transformation of the Hook.

        Args:
            a function specifying the transformation.
        Returns:
            the transformed Hook.
        """
        return f(self)


class Hook(AbstractHook):
    """Wrapper for callable taking a Trainer as input."""

    def __init__(self, wrapped: Callable[[p.TrainerProtocol], None]) -> None:
        """
        Args:
            wrapped: the function to be conditionally called.
        """
        self.wrapped = wrapped

    def __call__(self, trainer: p.TrainerProtocol) -> None:
        """
        Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
        self.wrapped(trainer)


class StaticHook(AbstractHook):
    """Ignoring arguments and execute a wrapped function."""

    def __init__(self, wrapped: Callable[[], None]):
        """
        Args:
            wrapped: the function to be wrapped and called statically.
        """
        self.wrapped = wrapped

    def __call__(self, trainer: p.TrainerProtocol) -> None:
        """
        Execute the call.

        Args:
            trainer: not used.
        """
        return self.wrapped()


class OptionalCallable(Hook, metaclass=abc.ABCMeta):
    """Abstract class for callables that execute based on custom conditions."""

    def __call__(self, trainer: p.TrainerProtocol) -> None:
        """
        Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
        if self._should_call(trainer):
            return self.wrapped(trainer)

        return None

    @abc.abstractmethod
    def _should_call(self, trainer: p.TrainerProtocol) -> bool:
        """Determine if the callable should be executed."""


class CallEvery(OptionalCallable):
    """Call a function at specified intervals."""

    def __init__(self,
                 wrapped: Callable[[p.TrainerProtocol], None],
                 interval: int,
                 start: int) -> None:
        """
        Args:
            start: the epoch to start calling the hook.
            interval: the frequency of calling the hook.
            wrapped: the function to be called periodically.
        """
        self.start = start
        self.interval = interval
        super().__init__(wrapped)
        return

    @override
    def _should_call(self, trainer: p.TrainerProtocol) -> bool:
        """
        Determine if the hook should be called based on epoch.

        Args:
            trainer: the trainer object to check for call condition.
        """
        epoch = trainer.model.epoch
        if epoch < self.start:
            return False

        return not (epoch - self.start) % self.interval or trainer.terminated


def call_every(
        interval: int,
        start: int = 0,
) -> Callable[[Callable[[p.TrainerProtocol], None]], CallEvery]:
    """
    Create a decorator for periodic hook execution.

    Args:
        start: the epoch to start calling the hook.
        interval: the frequency of calling the hook.

    Returns:
        A decorator that wraps a function in a CallEvery hook.
    """

    def _decorator(func: Callable[[p.TrainerProtocol], None]) -> CallEvery:
        return CallEvery(func, interval, start)

    return _decorator


@Hook
def saving_hook(trainer: p.TrainerProtocol) -> None:
    """
    Create a hook that saves the model's checkpoint.

    Args:
        The trainer instance.
    """
    trainer.save_checkpoint()
    return


def static_hook_class(
        cls: Callable[_P, Callable[[], None]]
) -> Callable[_P, StaticHook]:
    """
    Class decorator to wrap a callable class into a static hook type.

    Args:
        cls: a callable class that takes no arguments and returns None.

    Returns:
        A class that can be instantiated in the same way to have a static hook.
    """

    class StaticHookDecorator(StaticHook):

        def __init__(self, *args: _P.args, **kwargs: _P.kwargs):
            super().__init__(cls(*args, **kwargs))

    return StaticHookDecorator


class MetricMonitor:
    """
    Handle metric monitoring and alerts when performance stops increasing.

    Attributes:
        metric_name: name of the metric to monitor.
        optional_monitor: evaluation protocol to monitor.
        min_delta: minimum change required to qualify as an improvement.
        patience: number of checks to wait before triggering callback.
        filter: function to aggregate recent metric values.
        history: logs of the recorded metrics.
    """

    def __init__(
            self,
            metric: Optional[str | p.ObjectiveProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """
        Args:
            metric: name of the metric to monitor or metric calculator instance.
                Defaults to first metric found.
            monitor: evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: minimum change required to qualify as an improvement.
            patience: number of checks to wait before triggering callback.
            best_is: whether higher or lower metric values are better.
                Default 'auto' will determine it from initial measurements.
            filter_fn: function to aggregate recent metric values. Default
                gets the last value.
        """
        if metric is None or isinstance(metric, str):
            self.metric_name = metric
        elif name := getattr(metric, 'name', False):
            self.metric_name = str(name)
        else:
            self.metric_name = metric.__class__.__name__

        higher_is_better = getattr(metric, 'higher_is_better', None)
        if higher_is_better:
            self.best_is = 'higher'
        elif higher_is_better is False:
            self.best_is = 'lower'
        else:
            self.best_is = best_is

        if patience < 0:
            raise ValueError('Patience must be a non-negative integer.')

        self.filter = filter_fn
        self.min_delta = min_delta
        self.patience = patience
        self.optional_monitor = monitor
        self.history = list[float]()
        self._patience_countdown = patience
        self._best_value: Optional[float] = None
        return

    @property
    def best_value(self) -> float:
        """
        Get the best result observed so far.

        Returns:
            The best filtered value according to best_is criterion.

        Raises:
            ResultNotAvailableError: if no results have been logged yet.
        """
        if self._best_value is None:
            try:
                self._best_value = self.history[0]
            except IndexError:
                raise exceptions.ResultNotAvailableError()

        return self._best_value

    @best_value.setter
    def best_value(self, value: float) -> None:
        """Set the best result value."""
        self._best_value = value
        return

    @property
    def filtered_value(self) -> float:
        """
        Get the current value.

        Returns:
            The current value aggregated from recent ones.

        Raises:
            ResultNotAvailableError: if no results have been logged yet.
        """
        return self.filter(self.history)

    def is_better(self, value: float, reference: float) -> bool:
        """
        Determine if value is better than a reference value.

        When best_is is in 'auto' mode, it is assumed that the given value is
        better than the first recorded one.

        Args:
            value: the value to compare.
            reference: the reference.

        Returns:
            True if value is a potential improvement, False otherwise.
        """
        if value != value:  # Check for NaN
            return False

        if self.best_is == 'auto':
            if len(self.history) < 2:
                return True
            if self.history[0] > self.history[1]:
                self.best_is = 'lower'
            else:
                self.best_is = 'higher'

        if self.best_is == 'lower':
            return reference - self.min_delta > value
        else:
            return reference + self.min_delta < value

    def is_improving(self) -> bool:
        """
        Determine if the model performance is improving.

        Returns:
            True if there has been an improvement, False otherwise.

        Side Effects:
            If there is no improvement the patience countdown is reduced.
            Otherwise, it is restored to the maximum.
        """

        if len(self.history) <= 1:
            return True

        aggregated_value = self.filtered_value

        if self.is_better(aggregated_value, self.best_value):
            self.best_value = aggregated_value
            self._patience_countdown = self.patience
            return True

        self._patience_countdown -= 1
        return False

    def is_patient(self) -> bool:
        """Check whether to be patient."""
        return self._patience_countdown > 0

    def record_metric_value(self, instance: p.TrainerProtocol) -> None:
        """
        Register a new metric value from a monitored evaluation.

        If no evaluation is specified, it falls back on the trainer instance
        or if present, the validation instance contained there.

        Args:
            instance: Trainer instance to fall back on.

        Raises:
            MetricNotFoundError: if the specified metric is not found.
        """
        monitor = self._get_monitor(instance)
        last_metrics = metrics.repr_metrics(monitor.objective)

        if self.metric_name is None:
            self.metric_name = list(last_metrics.keys())[0]
        elif self.metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(monitor.name,
                                                 self.metric_name)

        value = last_metrics[self.metric_name]
        self.history.append(value)
        return

    def _get_monitor(self, instance: p.TrainerProtocol) -> p.EvaluationProtocol:
        if self.optional_monitor is None:
            if instance.validation is None:
                return cast(p.EvaluationProtocol, instance)  # correct

            return instance.validation

        return self.optional_monitor


class EarlyStoppingCallback:
    """
    Implement early stopping logic for training models.

    Attributes:
        monitor: monitor instance.
        start_from_epoch: start from epoch.
    """

    def __init__(
            self,
            metric: Optional[str | p.ObjectiveProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            filter_fn: Callable[[Sequence[float]], float] = get_last,
            start_from_epoch: int = 2,
    ) -> None:
        """
        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to first metric found.
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
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.start_from_epoch = start_from_epoch
        return

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Evaluate whether training should be stopped early.

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


class PruneCallback:
    """
    Implement pruning logic for training models.

    Attributes:
        monitor: monitor instance.
        thresholds: dictionary mapping epochs to pruning thresholds.
    """

    def __init__(
            self,
            thresholds: Mapping[int, float | None],
            metric: Optional[str | p.ObjectiveProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """
        Args:
            thresholds: dictionary mapping epochs to pruning values.
            metric: name of metric to monitor or metric calculator instance.
                Defaults to first metric found.
            monitor: evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: minimum change required to qualify as an improvement.
            best_is: whether higher or lower metric values are better.
               Default 'auto' will determine this from initial measurements.
            filter_fn: function to aggregate the intermediate results
            values. Default
                gets the last value.
        """
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=0,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.thresholds = thresholds
        self.trial_values: dict[int, float] = {}
        return

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Evaluate whether training should be stopped early.

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


class ChangeSchedulerOnPlateauCallback(metaclass=abc.ABCMeta):
    """
    Change learning rate schedule when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
    """

    def __init__(
            self,
            metric: Optional[str | p.ObjectiveProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 0,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            filter_fn: Callable[[Sequence[float]], float] = get_last,
            cooldown: int = 0,
    ) -> None:
        """
        Initialize the learning rate reduction callback.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to first metric found.
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
        self.monitor = MetricMonitor(
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

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Check if learning rate should be reduced and apply reduction if needed.

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

        scheduler = self.get_scheduler(epoch,
                                       instance.learning_scheme.scheduler)
        instance.update_learning_rate(base_lr=None, scheduler=scheduler)
        self._cooldown_counter = self.cooldown  # start cooldown period
        return

    @abc.abstractmethod
    def get_scheduler(self,
                      epoch: int,
                      scheduler: p.SchedulerProtocol) -> p.SchedulerProtocol:
        """
        Modify input scheduler.

        Args:
            epoch: current epoch.
            scheduler: scheduler to be modified.
        Returns:
            Modified scheduler.
        """


class ReduceLROnPlateau(ChangeSchedulerOnPlateauCallback):
    """
    Reduce the learning rate when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
        factor: factor by which to reduce the learning rate.
    """

    def __init__(
            self,
            metric: Optional[str | p.ObjectiveProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 0,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            filter_fn: Callable[[Sequence[float]], float] = get_last,
            factor: float = 0.1,
            cooldown: int = 0,
    ) -> None:
        """
        Initialize the learning rate reduction callback.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to first metric found.
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

    def get_scheduler(self,
                      epoch: int,
                      scheduler: p.SchedulerProtocol) -> p.SchedulerProtocol:
        """
        Modify input scheduler to scale down the learning rate.

        Args:
            epoch: not used.
            scheduler: scheduler to be modified.
        Returns:
            Modified scheduler.
        """
        return schedulers.RescaleScheduler(scheduler, self.factor)


class RestartScheduleOnPlateau(ChangeSchedulerOnPlateauCallback):
    """
    Restart the scheduling after plateauing.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
    """

    def get_scheduler(self,
                      epoch: int,
                      scheduler: p.SchedulerProtocol) -> p.SchedulerProtocol:
        """
        Consider training until now a warm-up and restart scheduling.

        Args:
            epoch: current epoch.
            scheduler: scheduler to be modified.
        Returns:
            Modified scheduler.
        """
        return schedulers.WarmupScheduler(scheduler, epoch)

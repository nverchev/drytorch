"""Module containing registry, callbacks, and hooks for a Trainer."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
import operator
from typing import Generic, Literal, Optional, ParamSpec, TypeVar, cast
from typing_extensions import override

from dry_torch import exceptions
from dry_torch import metrics
from dry_torch import protocols as p
from dry_torch import schedulers

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
             f: Callable[[Callable[[p.TrainerProtocol], None]], Hook],
             ) -> Hook:
        """
        Allow composition of callables.

        Args:
            f: a function specifying a change.
        Return:
            a wrapper around the class implementing the change.
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
        average_fn: function to average recent metric values.
    """

    def __init__(
            self,
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            average_fn: Callable[[Sequence[float]], float] = get_last,
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
            average_fn: function to aggregate recent metric values. Default
                gets the last value.
        """
        if metric is None or isinstance(metric, str):
            self.metric_name = metric
        elif name := getattr(metric, 'name', False):
            self.metric_name = str(name)
        else:
            self.metric_name = metric.__class__.__name__

        higher_is_better = getattr(metric, 'higher_is_better', None)
        if higher_is_better is True:
            self.best_is = 'higher'
        elif higher_is_better is False:
            self.best_is = 'lower'
        else:
            self.best_is = best_is

        if patience < 0:
            raise ValueError('Patience must be a non-negative integer.')

        self.average_fn = average_fn
        self.min_delta = min_delta
        self.patience = patience
        self.optional_monitor = monitor
        self._patience_countdown = patience
        self._best_result: Optional[float] = None
        self._monitor_log = list[float]()
        return

    @property
    def best_result(self) -> float:
        """
        Get the best result observed so far.

        Returns:
            The best metric value according to best_is criterion.

        Raises:
            ResultNotAvailableError: if no results have been logged yet.
        """
        if self._best_result is None:
            try:
                first_result = self._monitor_log[0]
            except IndexError:
                raise exceptions.ResultNotAvailableError()

            self._best_result = first_result
            return first_result

        return self._best_result

    @best_result.setter
    def best_result(self, value: float) -> None:
        """Set the best result value."""
        self._best_result = value
        return

    def is_best(self, value: float) -> bool:
        """
        Determine if value is better than recent performances.

        Args:
            value: the value to compare.

        Returns:
            True if value is a potential improvement, False otherwise.
        """
        if value != value:  # Check for NaN
            return False

        if self.best_is == 'auto':
            if self._monitor_log[0] > value:
                self.best_is = 'lower'
            else:
                self.best_is = 'higher'
            return True

        elif self.best_is == 'lower':
            return self.best_result - self.min_delta > value
        else:
            return self.best_result + self.min_delta < value

    def is_improving(self) -> bool:
        """
        Determine if the model performance is improving.

        Returns:
            True if there has been an improvement, False otherwise.

        Side Effects:
            If there is no improvement the patience countdown is reduced.
            Otherwise, it is restored to the maximum.
        """

        if not self._monitor_log:
            return True

        current_result = self.average_fn(self._monitor_log)

        if self.is_best(current_result):
            self.best_result = current_result
            self._patience_countdown = self.patience
            return True

        self._patience_countdown -= 1
        return False

    def is_patient(self) -> bool:
        """Check whether to be patient."""
        return self._patience_countdown > 0

    def register_metric(self, instance: p.TrainerProtocol) -> None:
        """
        Register new metric.

        Args:
            instance: Trainer instance.

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
        self._monitor_log.append(value)
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
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Callable[[Sequence[float]], float] = get_last,
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
            aggregate_fn: function to aggregate recent metric values. Default
                gets the last value.
            start_from_epoch: first epoch to start monitoring from.
        """
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            average_fn=aggregate_fn,
        )
        self.start_from_epoch = start_from_epoch
        return

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Evaluate whether training should be stopped early.

        Args:
            instance: Trainer instance to evaluate.
        """
        self.monitor.register_metric(instance)
        epoch = instance.model.epoch
        if epoch < self.start_from_epoch:
            return

        if self.monitor.is_improving() or self.monitor.is_patient():
            return

        best_result = self.monitor.best_result
        metric_name = self.monitor.metric_name
        instance.terminate_training(
            f'Training stopped with best result={best_result} {metric_name}.'
        )
        return


class PruneCallback:
    """
    Implement pruning logic for training models.

    Attributes:
        monitor: monitor instance.
        pruning: dictionary mapping epochs to pruning thresholds.
    """

    def __init__(
            self,
            pruning: dict[int, float],
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """
        Args:
            pruning: dictionary mapping epochs to pruning thresholds.
            metric: name of metric to monitor or metric calculator instance.
                Defaults to first metric found.
            monitor: evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: minimum change required to qualify as an improvement.
            best_is: whether higher or lower metric values are better.
               Default 'auto' will determine this from initial measurements.
            aggregate_fn: function to aggregate recent metric values. Default
                gets the last value.
        """
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=0,
            best_is=best_is,
            average_fn=aggregate_fn,
        )
        self.pruning = pruning
        return

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Evaluate whether training should be stopped early.

        Args:
            instance: trainer instance to evaluate.
        """
        epoch = instance.model.epoch
        if epoch not in self.pruning:
            return

        threshold = self.pruning[epoch]
        self.monitor.register_metric(instance)
        if self.monitor.is_best(threshold):
            metric_name = self.monitor.metric_name
            instance.terminate_training(
                f'Training stopped at {threshold=} {metric_name}.'
            )

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
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 0,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Callable[[Sequence[float]], float] = get_last,
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
            aggregate_fn: function to aggregate recent metric values. Default
                gets the last value.
            cooldown: calls to skip after changing the schedule.
        """
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            average_fn=aggregate_fn,
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
        epoch = instance.model.epoch

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        self.monitor.register_metric(instance)
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
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 0,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Callable[[Sequence[float]], float] = get_last,
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
            aggregate_fn: function to aggregate recent metric values. Default
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
            aggregate_fn=aggregate_fn,
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
        return schedulers.FactorScheduler(self.factor, scheduler)


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
        return schedulers.WarmupScheduler(epoch, scheduler)

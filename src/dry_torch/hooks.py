"""Registry and hooks for a class following the Trainer protocol."""

import abc
from collections import deque
from collections.abc import Callable, Sequence
import functools
import operator
from typing import Generic, Literal, Optional, ParamSpec, TypeAlias, TypeVar

from src.dry_torch import calculating
from src.dry_torch import log_events
from src.dry_torch import exceptions
from src.dry_torch import protocols as p
from src.dry_torch import schedulers

_T = TypeVar('_T')
_P = ParamSpec('_P')
_Hook: TypeAlias = Callable[[p.TrainerProtocol], None]


class HookRegistry(Generic[_T]):
    """
    A registry for managing and executing hooks.

    The hooks have a generic object as input and can access it.

    Attributes:
        hooks: A list of registered hooks.
    """

    def __init__(self) -> None:
        """
        Initializes the HookRegistry with an empty list of hooks.
        """
        self.hooks: list[Callable[[_T], None]] = []

    def register(self, hook: Callable[[_T], None]) -> None:
        """
        Registers a single hook.

        Args:
            hook: The hook to register.
        """
        self.hooks.append(hook)
        return

    def register_all(self, hook_list: list[Callable[[_T], None]]) -> None:
        """
        Registers multiple hooks.

        Args:
            hook_list: The list of hooks to register.
        """
        for hook in hook_list:
            self.register(hook)
        return

    def execute(self, input_object: _T) -> None:
        """
        Executes the registered hooks in order of registration.

        Args:
            input_object: The input to pass to each hook.
        """
        for hook in self.hooks:
            hook(input_object)
        return


def saving_hook() -> _Hook:
    """
    Creates a hook that saves the model's checkpoint.

    Returns:
        A callable hook that saves a checkpoint.
    """

    def _call(instance: p.TrainerProtocol) -> None:
        instance.save_checkpoint()

    return _call


def static_hook(hook: Callable[[], None]) -> _Hook:
    """
    Wraps a static callable as a hook.

    Args:
        hook: The static callable to wrap.

    Returns:
        A hook that invokes the static callable.
    """

    @functools.wraps(hook)
    def _call(_: p.TrainerProtocol) -> None:
        hook()

    return _call


def static_hook_closure(static_closure: Callable[_P, Callable[[], None]]
                        ) -> Callable[_P, _Hook]:
    """
    Creates a hook from a static closure.

    Args:
        static_closure: The static closure to wrap.

    Returns:
        A hook that invokes the static callable.
    """

    @functools.wraps(static_closure)
    def _closure_hook(*args: _P.args, **kwargs: _P.kwargs) -> _Hook:
        static_callable = static_closure(*args, **kwargs)

        def _call(_: p.TrainerProtocol) -> None:
            nonlocal static_callable
            return static_callable()

        return _call

    return _closure_hook


def call_every(interval: int, hook: _Hook, start: int = 0) -> _Hook:
    """
    Creates a hook that executes at specified intervals.

    Args:
        interval: The interval (in epochs) at which the hook is called.
        hook: The hook to execute.
        start: The starting epoch offset. Defaults to 0.

    Returns:
        A callable hook.
    """

    @functools.wraps(hook)
    def _call(instance: p.TrainerProtocol) -> None:
        epoch = instance.model.epoch
        if epoch % interval == start or instance.terminated:
            hook(instance)

    return _call


class MetricMonitor:
    """
    Handles metric monitoring and alerts when performance stop increasing.

    Attributes:
        metric_name: Name of the metric to monitor.
        optional_monitor: Evaluation protocol to monitor.
        min_delta: Minimum change required to qualify as an improvement.
        patience: Number of checks to wait before triggering callback.
        aggregate_fn: Function to aggregate recent metric values.
    """

    def __init__(
            self,
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> None:
        """
        Args:
            metric: Name of the metric to monitor or metric calculator instance.
                Defaults to first metric found.
            monitor: Evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: Minimum change required to qualify as an improvement.
            patience: Number of checks to wait before triggering callback.
            best_is: Whether higher or lower metric values are better.
                'auto' will determine this from first measurements.
            aggregate_fn: Function to aggregate recent metric values.
                Defaults to min/max based on best_is.
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

        self.aggregate_fn: Callable[[Sequence[float]], float]
        if aggregate_fn is None:
            self.aggregate_fn = operator.itemgetter(-1)
        else:
            self.aggregate_fn = aggregate_fn

        self.min_delta = min_delta
        self.patience = patience
        self.optional_monitor = monitor
        self._patience_countdown = patience
        self._best_result: Optional[float] = None
        self._monitor_log = deque[float](maxlen=patience + 1)
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

    def register_metric(self, instance: p.TrainerProtocol) -> None:
        """
        Register new metric.

        Args:
            instance: Trainer instance.

        Raises:
            MetricNotFoundError: If the specified metric is not found.
        """
        monitor = self._get_monitor(instance)
        last_metrics = calculating.repr_metrics(monitor.calculator)

        if self.metric_name is None:
            self.metric_name = list(last_metrics.keys())[0]
        elif self.metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(monitor.name,
                                                 self.metric_name)

        value = last_metrics[self.metric_name]
        self._monitor_log.append(value)
        return

    def is_patient(self) -> bool:
        """Check whether to be patient."""
        return self._patience_countdown > 1

    def is_improving(self) -> bool:
        """
        Determine if the model performance is improving.

        Returns:
            Whether there has been an improvement in the last {patience} calls.
        """

        if len(self._monitor_log) <= 1:
            return True

        aggregate_result = self.aggregate_fn(self._monitor_log)

        if self.is_best(aggregate_result):
            self.best_result = aggregate_result
            self._patience_countdown = self.patience
            return True
        self._patience_countdown -= 1
        return False

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

    def _get_monitor(self, instance: p.TrainerProtocol) -> p.EvaluationProtocol:
        if self.optional_monitor is None:
            if instance.validation is None:
                return instance  # type: ignore
            return instance.validation
        return self.optional_monitor


class EarlyStoppingCallback:
    """
    Implements early stopping logic for training models.

    Attributes:
        monitor: Monitor instance
        start_from_epoch: Start from epoch
    """

    def __init__(
            self,
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[Sequence[float]], float]] = None,
            start_from_epoch: int = 2,
    ) -> None:
        """
        Args:
            metric: Name of metric to monitor or metric calculator instance.
                            Defaults to first metric found.
            monitor: Evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: Minimum change required to qualify as an improvement.
            patience: Number of calls to wait before stopping.
                'auto' will determine this from first measurements.
            best_is: Whether higher or lower metric values are better.
            aggregate_fn: Function to aggregate recent metric values.
                Defaults to min/max based on best_is.
            start_from_epoch: First epoch to start monitoring from.
        """
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            aggregate_fn=aggregate_fn,
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
        instance.terminate_training()
        log_events.TerminatedTraining(epoch, 'early stopping')
        return


class PruneCallback:
    """
    Implements pruning logic for training models.

    Attributes:
        monitor: Monitor instance
        pruning: Dictionary mapping epochs to pruning thresholds
    """

    def __init__(
            self,
            pruning: dict[int, float],
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> None:
        """
        Args:
            pruning: Dictionary mapping epochs to pruning thresholds.
            metric: Name of metric to monitor or metric calculator instance.
                            Defaults to first metric found.
            monitor: Evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: Minimum change required to qualify as an improvement.
            best_is: Whether higher or lower metric values are better.
               'auto' will determine this from first measurements.
            aggregate_fn: Function to aggregate recent metric values.
                Defaults to min/max based on best_is.
        """
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=0,
            best_is=best_is,
            aggregate_fn=aggregate_fn,
        )
        self.pruning = pruning
        return

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Evaluate whether training should be stopped early.

        Args:
            instance: Trainer instance to evaluate.
        """
        epoch = instance.model.epoch
        if epoch not in self.pruning:
            return
        threshold = self.pruning[epoch]
        self.monitor.register_metric(instance)
        if self.monitor.is_best(threshold):
            instance.terminate_training()
            log_events.TerminatedTraining(epoch, 'pruning')
        return


class ChangeSchedulerOnPlateau(metaclass=abc.ABCMeta):
    """
    Changes learning rate schedule when a metric has stopped improving.

    Attributes:
        monitor: Monitor instance
        cooldown: Number of calls to skip after changing the schedule
    """

    def __init__(
            self,
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 0,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[Sequence[float]], float]] = None,
            cooldown: int = 0,
    ) -> None:
        """
        Initialize the learning rate reduction callback.

        Args:
            metric: Name of metric to monitor or metric calculator instance.
                Defaults to first metric found.
            monitor: Evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: Minimum change required to qualify as an improvement.
            patience: Number of checks to wait before changing the schedule.
            best_is: Whether higher or lower metric values are better.
                'auto' will determine this from first measurements.
            aggregate_fn: Function to aggregate recent metric values.
                Defaults to min/max based on best_is.
            cooldown: calls to skip after changing the schedule.
        """
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            aggregate_fn=aggregate_fn,
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
        instance.update_learning_rate(lr=None, scheduler=scheduler)

        # Start cooldown period
        self._cooldown_counter = self.cooldown
        return

    @abc.abstractmethod
    def get_scheduler(self,
                      epoch: int,
                      scheduler: p.SchedulerProtocol) -> p.SchedulerProtocol:
        """
        Modifies input scheduler.

        Args:
            epoch: Current epoch
            scheduler: Scheduler to be modified.
        """


class ReduceLROnPlateau(ChangeSchedulerOnPlateau):
    """
    Reduces learning rate when a metric has stopped improving.

    Attributes:
        monitor: Monitor instance
        cooldown: Number of calls to skip after changing the schedule
        factor: Factor by which to reduce the learning rate
    """

    def __init__(
            self,
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            patience: int = 0,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[Sequence[float]], float]] = None,
            factor: float = 0.1,
            cooldown: int = 0,
    ) -> None:
        """
        Initialize the learning rate reduction callback.

        Args:
            metric: Name of metric to monitor or metric calculator instance.
                Defaults to first metric found.
            monitor: Evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: Minimum change required to qualify as an improvement.
            patience: Number of checks to wait before changing the schedule.
            best_is: Whether higher or lower metric values are better.
                'auto' will determine this from first measurements.
            aggregate_fn: Function to aggregate recent metric values.
                Defaults to min/max based on best_is.
            cooldown: calls to skip after changing the schedule.
            factor: Factor by which to reduce the learning rate.
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
        Modifies input scheduler to scale down the learning rate.

        Args:
            epoch: Not used
            scheduler: Scheduler to be modified.
        """
        scaled_scheduler = schedulers.ConstantScheduler(self.factor)
        # composition is in reverse order such as in f \circle g
        return schedulers.CompositionScheduler(scheduler, scaled_scheduler)


class RestartScheduleOnPlateau(ChangeSchedulerOnPlateau):
    """
    Restarts the scheduling after plateauing.

    Attributes:
        monitor: Monitor instance
        cooldown: Number of calls to skip after changing the schedule
    """

    def get_scheduler(self,
                      epoch: int,
                      scheduler: p.SchedulerProtocol) -> p.SchedulerProtocol:
        """
        Consider training until now a warm-up and restart scheduling.

        Args:
            epoch: int
            scheduler: Scheduler to be modified.
        """
        return schedulers.WarmupScheduler(epoch, scheduler)

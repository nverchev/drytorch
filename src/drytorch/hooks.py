"""Module containing registry, callbacks, and hooks for a Trainer."""

from __future__ import annotations

import abc
import operator

from collections.abc import Callable, Mapping, Sequence
from typing import Generic, Literal, ParamSpec, TypeVar

from typing_extensions import override

from drytorch import objectives, schedulers
from drytorch.core import protocols as p


_T_contra = TypeVar('_T_contra', contravariant=True)
_P = ParamSpec('_P')
_Q = ParamSpec('_Q')
_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)
get_last = operator.itemgetter(-1)


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


class AbstractHook(Generic[_Input, _Target, _Output], metaclass=abc.ABCMeta):
    """Callable supporting bind operations."""

    @abc.abstractmethod
    def __call__(
        self, trainer: p.TrainerProtocol[_Input, _Target, _Output]
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """

    def bind(
        self,
        f: Callable[
            [AbstractHook[_Input, _Target, _Output]],
            AbstractHook[_Input, _Target, _Output],
        ],
        /,
    ) -> AbstractHook:
        """Allow transformation of the Hook.

        Args:
            f: a function specifying the transformation.

        Returns:
            the transformed Hook.
        """
        return f(self)


class Hook(AbstractHook[_Input, _Target, _Output]):
    """Wrapper for callable taking a Trainer as input."""

    def __init__(
        self,
        wrapped: Callable[[p.TrainerProtocol[_Input, _Target, _Output]], None],
    ) -> None:
        """Constructor.

        Args:
            wrapped: the function to be conditionally called.
        """
        self.wrapped = wrapped

    def __call__(
        self, trainer: p.TrainerProtocol[_Input, _Target, _Output]
    ) -> None:
        """Execute the call.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
        self.wrapped(trainer)


class StaticHook(AbstractHook[_Input, _Target, _Output]):
    """Ignoring arguments and execute a wrapped function."""

    def __init__(self, wrapped: Callable[[], None]):
        """Constructor.

        Args:
            wrapped: the function to be wrapped and called statically.
        """
        self.wrapped = wrapped

    def __call__(
        self, trainer: p.TrainerProtocol[_Input, _Target, _Output]
    ) -> None:
        """Execute the call.

        Args:
            trainer: not used.
        """
        return self.wrapped()


class OptionalCallable(Hook[_Input, _Target, _Output], metaclass=abc.ABCMeta):
    """Abstract class for callables that execute based on custom conditions."""

    def __call__(
        self, trainer: p.TrainerProtocol[_Input, _Target, _Output]
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
        self, trainer: p.TrainerProtocol[_Input, _Target, _Output]
    ) -> bool:
        """Determine if the callable should be executed."""


class CallEvery(OptionalCallable[_Input, _Target, _Output]):
    """Call a function at specified intervals."""

    def __init__(
        self,
        wrapped: Callable[[p.TrainerProtocol[_Input, _Target, _Output]], None],
        interval: int,
        start: int,
    ) -> None:
        """Constructor.

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
    def _should_call(
        self, trainer: p.TrainerProtocol[_Input, _Target, _Output]
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
    [Callable[[p.TrainerProtocol[_Input, _Target, _Output]], None]],
    CallEvery[_Input, _Target, _Output],
]:
    """Create a decorator for periodic hook execution.

    Args:
        start: the epoch to start calling the hook.
        interval: the frequency of calling the hook.

    Returns:
        A decorator that wraps a function in a CallEvery hook.
    """

    def _decorator(
        func: Callable[[p.TrainerProtocol[_Input, _Target, _Output]], None],
    ) -> CallEvery:
        return CallEvery(func, interval, start)

    return _decorator


@Hook
def saving_hook(trainer: p.TrainerProtocol[_Input, _Target, _Output]) -> None:
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


class EarlyStoppingCallback:
    """Implement early stopping logic for training models.

    Attributes:
        monitor: monitor instance.
        start_from_epoch: start from epoch.
    """

    def __init__(
        self,
        metric: str | p.ObjectiveProtocol | None = None,
        monitor: p.ValidationProtocol[_Input, _Target, _Output] | None = None,
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
        self.monitor = objectives.MetricMonitor(
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
        self, instance: p.TrainerProtocol[_Input, _Target, _Output]
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


class PruneCallback:
    """Implement pruning logic for training models.

    Attributes:
        monitor: monitor instance.
        thresholds: dictionary mapping epochs to pruning thresholds.
    """

    def __init__(
        self,
        thresholds: Mapping[int, float | None],
        metric: str | p.ObjectiveProtocol | None = None,
        monitor: p.ValidationProtocol[_Input, _Target, _Output] | None = None,
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
        self.monitor = objectives.MetricMonitor(
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

    def __call__(
        self, instance: p.TrainerProtocol[_Input, _Target, _Output]
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


class ChangeSchedulerOnPlateauCallback(metaclass=abc.ABCMeta):
    """Change the learning rate schedule when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
    """

    def __init__(
        self,
        metric: str | p.ObjectiveProtocol | None = None,
        monitor: p.ValidationProtocol[_Input, _Target, _Output] | None = None,
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
        self.monitor = objectives.MetricMonitor(
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
        self, instance: p.TrainerProtocol[_Input, _Target, _Output]
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
            epoch, instance.learning_scheme.scheduler
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


class ReduceLROnPlateau(ChangeSchedulerOnPlateauCallback):
    """Reduce the learning rate when a metric has stopped improving.

    Attributes:
        monitor: monitor instance.
        cooldown: number of calls to skip after changing the schedule.
        factor: factor by which to reduce the learning rate.
    """

    def __init__(
        self,
        metric: str | p.ObjectiveProtocol | None = None,
        monitor: p.ValidationProtocol[_Input, _Target, _Output] | None = None,
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


class RestartScheduleOnPlateau(ChangeSchedulerOnPlateauCallback):
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

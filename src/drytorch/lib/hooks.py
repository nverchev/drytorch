"""Hooks and callbacks for a Trainer."""

from __future__ import annotations

import abc
import functools

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Generic, Literal, ParamSpec, TypeVar, cast

from typing_extensions import override

from drytorch.core import protocols as p
from drytorch.lib import monitoring, schedulers
from drytorch.lib.schedulers import RestartScheduler


__all__ = [
    'CallEvery',
    'ChangeSchedulerOnPlateauCallback',
    'EarlyStoppingCallback',
    'Hook',
    'HookChain',
    'PruneCallback',
    'ReduceLROnPlateau',
    'RestartScheduleOnPlateau',
    'StaticHook',
    'TrainerHook',
    'call_every',
    'evaluation_hook',
    'saving_hook',
    'static_hook_class',
]

_T_contra = TypeVar('_T_contra', contravariant=True)
_P = ParamSpec('_P')
_Q = ParamSpec('_Q')
Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)


class HookChain(Generic[_T_contra]):
    """Execute a fixed chain of hooks in order.

    Attributes:
        hooks: the hooks in order of execution.
    """

    hooks: Final[tuple[Callable[[_T_contra], None], ...]]

    def __init__(
        self, hooks: Sequence[Callable[[_T_contra], None]] = ()
    ) -> None:
        """Initialize.

        Args:
            hooks: the hooks to execute, in order.
        """
        self.hooks = tuple(hooks)
        return

    def execute(self, input_object: _T_contra) -> None:
        """Execute the hooks in order.

        Args:
            input_object: the input to pass to each hook.
        """
        for hook in self.hooks:
            hook(input_object)

        return


class TrainerHook(
    abc.ABC,
    Generic[Input, Target, Output],
):
    """Compose control flow on side effects while tracking parameters."""

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
            Hook[Input, Target, Output],
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
        parameters = {
            **self._get_parameters(),
            **next_hook._get_parameters(),
        }
        return Hook(
            wrapped=next_hook.__call__,
            parameters=parameters,
            base_hook_name=self._get_name(),
        )

    @abc.abstractmethod
    def _get_name(self) -> str:
        """Return the name of the base hook for representation."""

    @abc.abstractmethod
    def _get_parameters(self) -> Mapping[str, Any]:
        """Return the metadata associated with the hook."""

    @staticmethod
    def _get_callable_name(wrapped: Callable[..., Any]) -> str:
        return getattr(wrapped, '__name__', wrapped.__class__.__name__)


class Hook(TrainerHook[Input, Target, Output]):
    """Wrapper for callable taking a Trainer as input.

    Attributes:
        base_hook_name: name of the base hook for representation.
        parameters: metadata associated with the hook.
    """

    base_hook_name: Final[str]
    parameters: Final[Mapping[str, Any]]
    _wrapped: Final[Callable[[p.TrainerProtocol[Input, Target, Output]], None]]

    def __init__(
        self,
        wrapped: Callable[[p.TrainerProtocol[Input, Target, Output]], None],
        base_hook_name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ):
        """Initialize.

        Args:
            wrapped: the function to be conditionally called.
            base_hook_name: name of the base hook for representation.
                Defaults to the name of the wrapped callable.
            parameters: metadata associated with the hook.
        """
        self._wrapped = wrapped
        if base_hook_name:
            self.base_hook_name = base_hook_name
        else:
            self.base_hook_name = self._get_callable_name(wrapped)

        self.parameters = {} if parameters is None else dict(parameters)
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

    @override
    def _get_name(self) -> str:
        return self.base_hook_name

    @override
    def _get_parameters(self) -> Mapping[str, Any]:
        return self.parameters


class StaticHook(TrainerHook[Any, Any, Any]):
    """Ignoring arguments and execute a wrapped function.

    Attributes:
        base_hook_name: name of the base hook for representation.
        parameters: metadata associated with the hook.
    """

    base_hook_name: Final[str]
    parameters: Final[Mapping[str, Any]]
    _wrapped: Final[Callable[[], None]]

    def __init__(
        self,
        wrapped: Callable[[], None],
        base_hook_name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ):
        """Initialize.

        Args:
            wrapped: the function to be wrapped and called statically.
            base_hook_name: name of the base hook for representation.
                Defaults to the name of the wrapped callable.
            parameters: metadata associated with the hook.
        """
        self._wrapped = wrapped
        if base_hook_name:
            self.base_hook_name = base_hook_name
        else:
            self.base_hook_name = self._get_callable_name(wrapped)

        self.parameters = {} if parameters is None else dict(parameters)
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

    @override
    def _get_name(self) -> str:
        return self.base_hook_name

    @override
    def _get_parameters(self) -> Mapping[str, Any]:
        return self.parameters


class CallEvery(Hook[Input, Target, Output]):
    """Metadata-aware wrapper for periodic execution.

    Attributes:
        base_hook_name: name of the base hook for representation.
        interval: the frequency of calling the function.
        parameters: metadata associated with the hook.
        start: the epoch to start calling the function.
    """

    interval: Final[int]
    start: Final[int]

    def __init__(
        self,
        wrapped: Callable[[p.TrainerProtocol[Input, Target, Output]], None],
        parameters: Mapping[str, Any] | None = None,
        interval: int = 1,
        start: int = 0,
    ):
        """Initialize.

        Args:
            wrapped: the function to be wrapped and called statically.
            parameters: metadata associated with the hook.
            interval: the frequency of calling the function.
            start: the epoch to start calling the function.

        Raises:
            ValueError: if interval is not positive or start is negative.
        """
        if interval <= 0:
            raise ValueError(f'interval must be positive, got {interval}')

        if start < 0:
            raise ValueError(f'start must be non-negative, got {start}')

        all_parameters = {
            **({} if parameters is None else parameters),
            'interval': interval,
            'start': start,
        }
        super().__init__(wrapped, '', all_parameters)
        self.interval = interval
        self.start = start
        return

    def __call__(
        self,
        trainer: p.TrainerProtocol[Input, Target, Output],
    ) -> None:
        """Optionally, calling wrapped callable.

        Args:
            trainer: the trainer to pass to the wrapped function.
        """
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


def evaluation_hook(
    evaluation: Callable[[], None], every: int = 1
) -> TrainerHook[Any, Any, Any]:
    """Create a hook running an evaluation every given number of epochs.

    With ``every`` above one, the evaluation also runs when training is
    terminated.

    Args:
        evaluation: the evaluation to run, for example a validation.
        every: the number of epochs between evaluations.

    Returns:
        The hook running the evaluation.

    Raises:
        ValueError: if every is not strictly positive.
    """
    if every < 1:
        raise ValueError(f'every must be positive. Got {every}.')

    hook = StaticHook(evaluation)
    if every > 1:
        return hook.bind(call_every(every))

    return hook


def static_hook_class(
    cls: Callable[_P, Callable[[], None]],
) -> Callable[_P, StaticHook]:
    """Class decorator to wrap a callable class into a static hook type.

    Args:
        cls: a callable class that takes no arguments and returns None.

    Returns:
        A class that can be instantiated in the same way to have a static hook.

    Raises:
        TypeError: if ``cls`` is not a class.
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


class EarlyStoppingCallback(Generic[Output, Target]):
    """Implement early stopping logic for training models.

    Attributes:
        monitor: monitor instance.
        start_from_epoch: epoch to start monitoring from.
    """

    monitor: Final[monitoring.MetricMonitor[Output, Target]]
    start_from_epoch: Final[int]

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        source: p.SourceProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 10,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = monitoring.get_last,
        start_from_epoch: int = 2,
    ) -> None:
        """Initialize.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            source: the source computing the metric. Defaults to the
                validation if available, the trainer otherwise.
            min_delta: minimum absolute change to qualify as an improvement.
            patience: number of calls to wait before stopping.
            best_is: whether higher or lower metric values are better.
                Default 'auto' takes it from the metric when it states its
                direction, from initial measurements otherwise.
            filter_fn: function to aggregate recent metric values. Default
                gets the last value.
            start_from_epoch: first epoch to start monitoring from.
        """
        self.monitor = monitoring.MetricMonitor(
            metric=metric,
            source=source,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.start_from_epoch = start_from_epoch
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
        trial_values: dictionary mapping epochs to recorded metric values.
    """

    monitor: Final[monitoring.MetricMonitor[Output, Target]]
    thresholds: Final[Mapping[int, float | None]]
    trial_values: dict[int, float]

    def __init__(
        self,
        thresholds: Mapping[int, float | None],
        metric: str | p.ObjectiveProtocol[Output, Target] | None = None,
        source: p.SourceProtocol | None = None,
        min_delta: float = 1e-8,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = monitoring.get_last,
    ) -> None:
        """Initialize.

        Args:
            thresholds: dictionary mapping epochs to pruning values.
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            source: the source computing the metric. Defaults to the
                validation if available, the trainer otherwise.
            min_delta: minimum absolute change to qualify as an improvement.
            best_is: whether higher or lower metric values are better.
                Default 'auto' takes it from the metric when it states its
                direction, from initial measurements otherwise.
            filter_fn: function to aggregate the intermediate results
            values. Default
                gets the last value.
        """
        self.monitor = monitoring.MetricMonitor(
            metric=metric,
            source=source,
            min_delta=min_delta,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.thresholds = thresholds
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
        self.trial_values[epoch] = value
        if threshold is not None:
            if not self.monitor.is_better(value, threshold):
                metric_name = self.monitor.metric_name
                msg = f'Training stopped at {threshold=} {metric_name}'
                instance.terminate_training(msg)

        return


class ChangeSchedulerOnPlateauCallback(abc.ABC, Generic[Output, Target]):
    """Change the learning rate schedule when a metric has stopped improving.

    Attributes:
        cooldown: number of calls to skip after changing the schedule.
        monitor: monitor instance.
    """

    cooldown: Final[int]
    monitor: Final[monitoring.MetricMonitor[Output, Target]]
    _cooldown_counter: int

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        source: p.SourceProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = monitoring.get_last,
        cooldown: int = 0,
    ) -> None:
        """Initialize.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            source: the source computing the metric. Defaults to the
                validation if available, the trainer otherwise.
            min_delta: minimum absolute change to qualify as an improvement.
            patience: number of checks to wait before changing the schedule.
            best_is: whether higher or lower metric values are better.
                Default 'auto' takes it from the metric when it states its
                direction, from initial measurements otherwise.
            filter_fn: function to aggregate recent metric values. Default
                gets the last value.
            cooldown: calls to skip after changing the schedule.
        """
        self.cooldown = cooldown
        self.monitor = monitoring.MetricMonitor(
            metric=metric,
            source=source,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self._cooldown_counter = 0
        return

    def __call__(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> None:
        """Check if there is a plateau and reduce the learning rate if needed.

        Args:
            instance: trainer instance to evaluate.
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
        cooldown: number of calls to skip after changing the schedule.
        factor: factor by which to reduce the learning rate.
        monitor: monitor instance.
    """

    factor: Final[float]
    _n_reductions: int

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        source: p.SourceProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = monitoring.get_last,
        factor: float = 0.1,
        cooldown: int = 0,
    ) -> None:
        """Initialize.

        Args:
            metric: name of metric to monitor or metric calculator instance.
                Defaults to the first metric found.
            source: the source computing the metric. Defaults to the
                validation if available, the trainer otherwise.
            min_delta: minimum absolute change to qualify as an improvement.
            patience: number of checks to wait before changing the schedule.
            best_is: whether higher or lower metric values are better.
                Default 'auto' takes it from the metric when it states its
                direction, from initial measurements otherwise.
            filter_fn: function to aggregate recent metric values. Default
                gets the last value.
            cooldown: calls to skip after changing the schedule.
            factor: factor by which to reduce the learning rate.
        """
        super().__init__(
            metric=metric,
            source=source,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
            cooldown=cooldown,
        )
        self.factor = factor
        self._n_reductions = 0
        return

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
        self._n_reductions += 1
        return schedulers.RescaleScheduler(
            scheduler, self.factor**self._n_reductions
        )


class RestartScheduleOnPlateau(
    ChangeSchedulerOnPlateauCallback[Output, Target]
):
    """Restart the scheduling after plateauing.

    Attributes:
        cooldown: number of calls to skip after changing the schedule.
        monitor: monitor instance.
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

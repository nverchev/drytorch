"""Hooks and callbacks for a Trainer."""

from __future__ import annotations

import abc

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, ParamSpec

from typing_extensions import override

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import monitoring, schedulers
from drytorch.lib.schedulers import RestartScheduler


__all__ = [
    'CallEvery',
    'ChangeSchedulerOnPlateauCallback',
    'EarlyStoppingCallback',
    'EvaluationHook',
    'HookChain',
    'PruneCallback',
    'ReduceLROnPlateau',
    'RestartScheduleOnPlateau',
    'SavingHook',
    'TrainerHook',
]

_P = ParamSpec('_P')


class HookChain:
    """Execute a fixed chain of hooks in order.

    Attributes:
        hooks: the hooks in order of execution.
    """

    hooks: Final[tuple[p.HookProtocol, ...]]

    def __init__(self, hooks: Sequence[p.HookProtocol] = ()) -> None:
        """Initialize.

        Args:
            hooks: the hooks to execute, in order.
        """
        self.hooks = tuple(hooks)
        return

    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Bind every hook of the chain to the trainer that calls them.

        Args:
            trainer: the trainer calling the hooks.
        """
        for hook in self.hooks:
            hook.bind_trainer(trainer)

        return

    def execute(self) -> None:
        """Execute the hooks in order."""
        for hook in self.hooks:
            hook()

        return


class TrainerHook(abc.ABC):
    """Hook bound to a trainer."""

    _trainer: p.TrainerProtocol[Any, Any, Any] | None

    def __init__(self) -> None:
        """Initialize."""
        self._trainer = None
        return

    @abc.abstractmethod
    def __call__(self) -> None:
        """Execute the hook."""

    @property
    def trainer(self) -> p.TrainerProtocol[Any, Any, Any]:
        """The trainer the hook is bound to.

        Raises:
            HookNotBoundError: if the trainer has not bound the hook yet.
        """
        if self._trainer is None:
            raise exceptions.HookNotBoundError(self._get_name())

        return self._trainer

    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Bind the hook to the trainer that calls it.

        Args:
            trainer: the trainer calling the hook.

        Raises:
            HookAlreadyBoundError: if another trainer already bound the hook.
        """
        if self._trainer is not None and self._trainer is not trainer:
            raise exceptions.HookAlreadyBoundError(self._get_name())

        self._trainer = trainer
        return

    def _get_name(self) -> str:
        return self.__class__.__name__

    def _get_parameters(self) -> Mapping[str, Any]:
        return {}


class EvaluationHook(TrainerHook):
    """Run an evaluation during the training.

    Attributes:
        evaluation: the evaluation to run, for example a validation.
    """

    evaluation: Final[p.SourceProtocol]

    def __init__(self, evaluation: p.SourceProtocol) -> None:
        """Initialize.

        Args:
            evaluation: the evaluation to run.
        """
        super().__init__()
        self.evaluation = evaluation
        return

    @override
    def __call__(self) -> None:
        """Run the evaluation."""
        self.evaluation()
        return


class SavingHook(TrainerHook):
    """Save the checkpoint of the model of its trainer."""

    @override
    def __call__(self) -> None:
        """Save the checkpoint."""
        self.trainer.save_checkpoint()
        return


class CallEvery(TrainerHook):
    """Call a hook periodically.

    Attributes:
        hook: the hook to call.
        interval: the frequency of calling the hook.
        start: the epoch to start calling the hook.
    """

    hook: Final[TrainerHook]
    interval: Final[int]
    start: Final[int]

    def __init__(
        self,
        hook: TrainerHook,
        interval: int = 1,
        start: int = 0,
    ) -> None:
        """Initialize.

        Args:
            hook: the hook to call periodically.
            interval: the frequency of calling the hook.
            start: the epoch to start calling the hook.

        Raises:
            ValueError: if interval is not positive or start is negative.
        """
        if interval <= 0:
            raise ValueError(f'interval must be positive, got {interval}')

        if start < 0:
            raise ValueError(f'start must be non-negative, got {start}')

        super().__init__()
        self.hook = hook
        self.interval = interval
        self.start = start
        return

    @override
    def __call__(self) -> None:
        """Call the hook when the epoch matches, or when training stopped."""
        epoch = self.trainer.model.epoch
        if epoch < self.start:
            return

        if not (epoch - self.start) % self.interval or self.trainer.terminated:
            self.hook()

        return

    @override
    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Bind the wrapped hook as well."""
        super().bind_trainer(trainer)
        self.hook.bind_trainer(trainer)
        return

    @override
    def _get_name(self) -> str:
        return self.hook._get_name()

    @override
    def _get_parameters(self) -> Mapping[str, Any]:
        return {
            **self.hook._get_parameters(),
            'interval': self.interval,
            'start': self.start,
        }


class EarlyStoppingCallback(TrainerHook):
    """Implement early stopping logic for training models.

    Attributes:
        monitor: follows the metric the callback acts on.
        start_from_epoch: epoch to start monitoring from.
    """

    monitor: Final[monitoring.MetricMonitor[Any, Any]]
    start_from_epoch: Final[int]

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Any, Any] | str | None = None,
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
        super().__init__()
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

    @override
    def __call__(self) -> None:
        """Evaluate whether training should be stopped early."""
        self.monitor.record_metric_value(self.trainer)
        epoch = self.trainer.model.epoch
        if epoch < self.start_from_epoch:
            return

        if self.monitor.is_improving() or self.monitor.is_patient():
            return

        best_result = self.monitor.best_value
        metric_name = self.monitor.metric_name
        msg = f'Training stopped with best result={best_result} {metric_name}'
        self.trainer.terminate_training(msg)
        return


class PruneCallback(TrainerHook):
    """Implement pruning logic for training models.

    Attributes:
        monitor: follows the metric the callback acts on.
        thresholds: dictionary mapping epochs to pruning thresholds.
        trial_values: dictionary mapping epochs to recorded metric values.
    """

    monitor: Final[monitoring.MetricMonitor[Any, Any]]
    thresholds: Final[Mapping[int, float | None]]
    trial_values: dict[int, float]

    def __init__(
        self,
        thresholds: Mapping[int, float | None],
        metric: str | p.ObjectiveProtocol[Any, Any] | None = None,
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
        super().__init__()
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

    @override
    def __call__(self) -> None:
        """Evaluate whether training should be stopped early."""
        self.monitor.record_metric_value(self.trainer)
        epoch = self.trainer.model.epoch
        if epoch not in self.thresholds:
            return

        threshold = self.thresholds[epoch]
        value = self.monitor.filtered_value
        self.trial_values[epoch] = value
        if threshold is not None:
            if not self.monitor.is_better(value, threshold):
                metric_name = self.monitor.metric_name
                msg = f'Training stopped at {threshold=} {metric_name}'
                self.trainer.terminate_training(msg)

        return


class ChangeSchedulerOnPlateauCallback(TrainerHook, abc.ABC):
    """Change the learning rate schedule when a metric has stopped improving.

    Attributes:
        cooldown: number of calls to skip after changing the schedule.
        monitor: follows the metric the callback acts on.
    """

    cooldown: Final[int]
    monitor: Final[monitoring.MetricMonitor[Any, Any]]
    _cooldown_counter: int

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Any, Any] | str | None = None,
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
        super().__init__()
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

    @override
    def __call__(self) -> None:
        """Check if there is a plateau and reduce the learning rate."""
        self.monitor.record_metric_value(self.trainer)
        epoch = self.trainer.model.epoch

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        if self.monitor.is_improving() or self.monitor.is_patient():
            return

        scheduler = self.get_scheduler(
            epoch, self.trainer.learning_schema.scheduler
        )
        self.trainer.update_learning_rate(base_lr=None, scheduler=scheduler)
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
        cooldown: number of calls to skip after changing the schedule.
        factor: factor by which to reduce the learning rate.
        monitor: follows the metric the callback acts on.
    """

    factor: Final[float]
    _n_reductions: int

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Any, Any] | str | None = None,
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


class RestartScheduleOnPlateau(ChangeSchedulerOnPlateauCallback):
    """Restart the scheduling after plateauing.

    Attributes:
        cooldown: number of calls to skip after changing the schedule.
        monitor: follows the metric the callback acts on.
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

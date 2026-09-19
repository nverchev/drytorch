"""Record of the metrics and conditions deciding when a hook runs."""

from __future__ import annotations

import abc
import dataclasses
import operator
import types

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Generic, Literal, Self, TypeVar

from typing_extensions import override

from drytorch.core import exceptions
from drytorch.core import protocols as p


__all__ = [
    'ALWAYS',
    'After',
    'Always',
    'AndCondition',
    'AtEpochs',
    'Condition',
    'Cooldown',
    'Every',
    'LogicalCondition',
    'MetricCondition',
    'Monitor',
    'NotCondition',
    'OnTermination',
    'OrCondition',
    'Plateau',
    'Watch',
    'WorseThan',
    'get_last',
]


get_last: Final = operator.itemgetter(-1)


@dataclasses.dataclass(frozen=True)
class Watch:
    """What a condition follows during the training.

    Attributes:
        best_is: whether higher or lower values are better. Default takes it
            from the objective computing the metric.
        filter_fn: aggregates the recent values. Default takes the last.
        metric: the metric to follow, by name or by objective. Default
            follows the objective of the source.
        source: the evaluation computing the metric. Default follows the
            training.
    """

    best_is: Literal['higher', 'lower'] | None = None
    filter_fn: Callable[[Sequence[float]], float] = get_last
    metric: p.ObjectiveProtocol[Any, Any] | str | None = None
    source: p.SourceProtocol | None = None

    def resolve(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> Monitor:
        """Create the record of the metric for a trainer.

        Args:
            trainer: the trainer whose metrics are followed.

        Returns:
            The record of the metric.

        Raises:
            ValueError: if the metric cannot be determined.
        """
        source = trainer if self.source is None else self.source
        objective = self._get_objective(source)
        return Monitor(
            source,
            self._get_metric_name(source, objective),
            best_is=self.best_is or self._get_best_is(objective),
            filter_fn=self.filter_fn,
        )

    def _get_best_is(
        self, objective: p.ObjectiveProtocol[Any, Any] | None
    ) -> Literal['higher', 'lower'] | None:
        if isinstance(objective, p.HigherIsBetterProtocol):
            higher_is_better = objective.higher_is_better
            if higher_is_better is not None:
                return 'higher' if higher_is_better else 'lower'

        return None

    def _get_metric_name(
        self,
        source: p.SourceProtocol,
        objective: p.ObjectiveProtocol[Any, Any] | None,
    ) -> str:
        if isinstance(self.metric, str):
            return self.metric

        if isinstance(objective, p.NamedProtocol) and objective.name:
            return str(objective.name)

        msg = f'Cannot tell which metric of {source.name} to follow.'
        raise ValueError(msg)

    def _get_objective(
        self, source: p.SourceProtocol
    ) -> p.ObjectiveProtocol[Any, Any] | None:
        if isinstance(self.metric, p.ObjectiveProtocol):
            return self.metric

        objective = source.objective
        if self.metric is None:
            return objective

        # the objective describes a named metric only when the names agree
        if isinstance(objective, p.NamedProtocol):
            if objective.name == self.metric:
                return objective

        return None


class Monitor:
    """Record of one metric of a source across the epochs.

    Instances come from :meth:`Watch.resolve`, which finds the source, the
    name of the metric and its direction.

    Attributes:
        best_is: whether higher or lower values are better, when known.
        filter_fn: aggregates the recent values into the current one.
        metric_name: the name of the followed metric.
        source: the evaluation computing the metric.
    """

    best_is: Final[Literal['higher', 'lower'] | None]
    filter_fn: Final[Callable[[Sequence[float]], float]]
    metric_name: Final[str]
    source: Final[p.SourceProtocol]
    _first_observed_epoch: int | None
    _values: Final[dict[int, float]]

    def __init__(
        self,
        source: p.SourceProtocol,
        metric_name: str,
        best_is: Literal['higher', 'lower'] | None = None,
        filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """Initialize.

        Args:
            source: the evaluation computing the metric.
            metric_name: the name of the metric to follow.
            best_is: whether higher or lower values are better.
            filter_fn: aggregates the recent values. Default takes the last.
        """
        self.best_is = best_is
        self.filter_fn = filter_fn
        self.metric_name = metric_name
        self.source = source
        self._first_observed_epoch = None
        self._values = {}
        return

    @property
    def epochs(self) -> Sequence[int]:
        """The epochs of the recorded values."""
        return tuple(self._values)

    @property
    def history(self) -> Sequence[float]:
        """The recorded values, in the order they were recorded."""
        return tuple(self._values.values())

    @property
    def value(self) -> float:
        """The current value, aggregated from the recent ones.

        Raises:
            ResultNotAvailableError: if no value has been recorded yet.
        """
        if not self._values:
            raise exceptions.ResultNotAvailableError()

        return self.filter_fn(self.history)

    @property
    def values(self) -> Mapping[int, float]:
        """The recorded values, by epoch."""
        return types.MappingProxyType(self._values)

    def is_better(
        self, value: float, reference: float, min_delta: float = 0.0
    ) -> bool:
        """Check whether a value is better than a reference.

        Args:
            value: the value to compare.
            reference: the value to compare against.
            min_delta: the minimum change that counts as better.

        Returns:
            ``True`` if the value is better than the reference.

        Raises:
            ValueError: if the direction of the metric is unknown.
        """
        if self.best_is is None:
            msg = f'The direction of {self.metric_name} is unknown.'
            raise ValueError(msg)

        if value != value:  # check for NaN
            return False

        if self.best_is == 'lower':
            return reference - min_delta > value

        return reference + min_delta < value

    def record(self, epoch: int) -> None:
        """Record the value the source computed for the epoch.

        Nothing is recorded when the source did not run for this epoch, so a
        source running every few epochs leaves no stale value behind.

        Args:
            epoch: the epoch of the value.

        Raises:
            MetricNotFoundError: if the source computed other metrics.
            SourceRunAfterHookError: if the source ran after the hook in an
                observed epoch.
        """
        if self._first_observed_epoch is None:
            self._first_observed_epoch = epoch

        if epoch in self._values:
            return

        computed_metrics = self.source.computed_metrics
        if not computed_metrics:
            last_run = self.source.last_run_epoch
            if (
                last_run > 0
                and self._first_observed_epoch <= last_run
                and last_run not in self._values
            ):
                raise exceptions.SourceRunAfterHookError(
                    self.source.name, last_run
                )

            return

        if self.metric_name not in computed_metrics:
            raise exceptions.MetricNotFoundError(
                self.source.name, self.metric_name
            )

        self._values[epoch] = computed_metrics[self.metric_name]
        return


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


class Condition(abc.ABC):
    """Decide whether a hook runs when its trainer calls it."""

    @abc.abstractmethod
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Check the condition.

        Args:
            hook: the hook asking whether to run.

        Returns:
            ``True`` if the hook should run.
        """

    def __and__(self, other: _C) -> AndCondition[Self, _C]:
        """Hold when both conditions hold.

        Args:
            other: the condition to combine with.

        Returns:
            The combined condition.
        """
        return AndCondition(self, other)

    def __invert__(self) -> NotCondition[Self]:
        """Hold when the condition does not.

        Returns:
            The negated condition.
        """
        return NotCondition(self)

    def __or__(self, other: _C) -> OrCondition[Self, _C]:
        """Hold when either condition holds.

        Args:
            other: the condition to combine with.

        Returns:
            The combined condition.
        """
        return OrCondition(self, other)

    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Bind the condition to the trainer calling the hook.

        Args:
            trainer: the trainer calling the hook.
        """
        _not_used = trainer
        return

    def observe(self, hook: p.HookProtocol) -> None:
        """Record what the condition follows, on every epoch.

        Args:
            hook: the hook the condition belongs to.
        """
        _not_used = hook
        return


_C = TypeVar('_C', bound=Condition)
_C_co = TypeVar('_C_co', bound=Condition, covariant=True)
_L_co = TypeVar('_L_co', bound=Condition, covariant=True)
_R_co = TypeVar('_R_co', bound=Condition, covariant=True)


class Always(Condition):
    """Hold on every call."""

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold on every call."""
        _not_used = hook
        return True


ALWAYS: Final = Always()


class After(Condition):
    """Hold from a given epoch on.

    Attributes:
        epoch: the first epoch when the condition holds.
    """

    epoch: Final[int]

    def __init__(self, epoch: int) -> None:
        """Initialize.

        Args:
            epoch: the first epoch when the condition holds.
        """
        self.epoch = epoch
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the model reached the epoch."""
        return hook.trainer.model.epoch >= self.epoch


class AtEpochs(Condition):
    """Hold at given epochs.

    Attributes:
        epochs: the epochs when the condition holds.
    """

    epochs: Final[frozenset[int]]

    def __init__(self, epochs: Sequence[int] | frozenset[int]) -> None:
        """Initialize.

        Args:
            epochs: the epochs when the condition holds.
        """
        self.epochs = frozenset(epochs)
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the epoch is one of the given ones."""
        return hook.trainer.model.epoch in self.epochs


class Cooldown(Condition):
    """Hold when the hook has not run for a number of epochs.

    Attributes:
        epochs: the epochs to skip after the hook ran.
    """

    epochs: Final[int]

    def __init__(self, epochs: int) -> None:
        """Initialize.

        Args:
            epochs: the epochs to skip after the hook ran.

        Raises:
            ValueError: if the number of epochs is negative.
        """
        if epochs < 0:
            raise ValueError(f'epochs must be non-negative, got {epochs}')

        self.epochs = epochs
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the hook last ran long enough ago."""
        last_epoch = hook.last_run_epoch
        if last_epoch == 0:  # the hook never ran
            return True

        return hook.trainer.model.epoch - last_epoch > self.epochs


class Every(Condition):
    """Hold periodically, and when the training is terminated.

    Attributes:
        interval: the number of epochs between two calls.
        start: the epoch from which the hook runs.
    """

    interval: Final[int]
    start: Final[int]

    def __init__(self, interval: int, start: int = 0) -> None:
        """Initialize.

        Args:
            interval: the number of epochs between two calls.
            start: the epoch from which the hook runs.

        Raises:
            ValueError: if interval is not positive or start is negative.
        """
        if interval <= 0:
            raise ValueError(f'interval must be positive, got {interval}')

        if start < 0:
            raise ValueError(f'start must be non-negative, got {start}')

        self.interval = interval
        self.start = start
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold every interval epochs, or when the training is terminated."""
        epoch = hook.trainer.model.epoch
        if epoch < self.start:
            return False

        if hook.trainer.terminated:
            return True

        return not (epoch - self.start) % self.interval


class OnTermination(Condition):
    """Hold when the training has been terminated."""

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the trainer is terminated."""
        return hook.trainer.terminated


class MetricCondition(Condition, abc.ABC):
    """Base class for conditions following a metric.

    Subclass it to decide on the values of a metric: the condition resolves
    what it follows when the trainer binds it, and records on every call.

    Attributes:
        watch: what the condition follows.
    """

    watch: Final[Watch]
    _monitor: Monitor | None

    def __init__(self, watch: Watch | None = None) -> None:
        """Initialize.

        Args:
            watch: what to follow. Default follows the loss of the trainer.
        """
        self.watch = Watch() if watch is None else watch
        self._monitor = None
        return

    @property
    def monitor(self) -> Monitor:
        """The record of the followed metric.

        Raises:
            ResultNotAvailableError: if no trainer bound the condition yet.
        """
        if self._monitor is None:
            raise exceptions.ResultNotAvailableError()

        return self._monitor

    @override
    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Create the record of the metric for the trainer.

        Raises:
            ValueError: if the metric or its direction cannot be determined.
        """
        monitor = self.watch.resolve(trainer)
        if monitor.best_is is None:
            msg = (
                f'The direction of {monitor.metric_name} is unknown:'
                ' specify best_is.'
            )
            raise ValueError(msg)

        self._monitor = monitor
        return

    @override
    def observe(self, hook: p.HookProtocol) -> None:
        """Record the value of the current epoch."""
        self.monitor.record(hook.trainer.model.epoch)
        return


class Plateau(MetricCondition):
    """Hold when the metric stopped improving.

    Attributes:
        min_delta: the minimum change counting as an improvement.
        patience: the number of records to wait for an improvement.
        watch: what the condition follows.

    A record happens when the source computes the metric, so a source
    running every few epochs makes the patience count its runs, not epochs.
    """

    min_delta: Final[float]
    patience: Final[int]
    _best_value: float | None
    _since_improvement: int

    def __init__(
        self,
        patience: int = 0,
        min_delta: float = 1e-8,
        watch: Watch | None = None,
    ) -> None:
        """Initialize.

        Args:
            patience: the number of records to wait for an improvement, as
                many as the runs of the source rather than epochs.
            min_delta: the minimum change counting as an improvement.
            watch: what to follow. Default follows the loss of the trainer.

        Raises:
            ValueError: if patience is negative.
        """
        if patience < 0:
            raise ValueError(f'patience must be non-negative, got {patience}')

        super().__init__(watch)
        self.min_delta = min_delta
        self.patience = patience
        self._best_value = None
        self._since_improvement = 0
        return

    @property
    def best_value(self) -> float:
        """The best value observed so far.

        Raises:
            ResultNotAvailableError: if no value has been observed yet.
        """
        if self._best_value is None:
            raise exceptions.ResultNotAvailableError()

        return self._best_value

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the last values did not improve on the previous ones."""
        if hook.trainer.model.epoch not in self.monitor.values:
            return False

        return self._since_improvement >= max(self.patience, 1)

    @override
    def observe(self, hook: p.HookProtocol) -> None:
        """Record the value of the current epoch and compare it."""
        recorded = len(self.monitor.history)
        super().observe(hook)
        if len(self.monitor.history) == recorded:  # nothing new to compare
            return

        value = self.monitor.value
        if self._best_value is None:
            self._best_value = value
            return

        if self.monitor.is_better(value, self._best_value, self.min_delta):
            self._best_value = value
            self._since_improvement = 0
        else:
            self._since_improvement += 1

        return


class WorseThan(MetricCondition):
    """Hold when the metric is worse than a threshold at given epochs.

    Attributes:
        thresholds: the value to reach at each given epoch.
        watch: what the condition follows.
    """

    thresholds: Final[Mapping[int, float | None]]

    def __init__(
        self,
        thresholds: Mapping[int, float | None],
        watch: Watch | None = None,
    ) -> None:
        """Initialize.

        Args:
            thresholds: the value to reach at each given epoch.
            watch: what to follow. Default follows the loss of the trainer.
        """
        super().__init__(watch)
        self.thresholds = thresholds
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the value does not reach the threshold of the epoch.

        Raises:
            ResultNotAvailableError: if the threshold epoch has no record.
        """
        epoch = hook.trainer.model.epoch
        if epoch not in self.thresholds:
            return False

        threshold = self.thresholds[epoch]
        if threshold is None:
            return False

        if epoch not in self.monitor.values:
            raise exceptions.ResultNotAvailableError()

        return not self.monitor.is_better(self.monitor.values[epoch], threshold)


class LogicalCondition(Condition, abc.ABC):
    """Base class for conditions combining other conditions."""

    @property
    @abc.abstractmethod
    def operands(self) -> tuple[Condition, ...]:
        """The combined conditions."""

    @override
    def bind_trainer(self, trainer: p.TrainerProtocol[Any, Any, Any]) -> None:
        """Bind every combined condition."""
        for condition in self.operands:
            condition.bind_trainer(trainer)

        return

    @override
    def observe(self, hook: p.HookProtocol) -> None:
        """Let every combined condition observe what it follows."""
        for condition in self.operands:
            condition.observe(hook)

        return


class AndCondition(LogicalCondition, Generic[_L_co, _R_co]):
    """Hold when both conditions hold.

    Attributes:
        left: the condition checked first.
        right: the condition checked when the first holds.
    """

    left: Final[_L_co]
    right: Final[_R_co]

    def __init__(self, left: _L_co, right: _R_co) -> None:
        """Initialize.

        Args:
            left: the condition checked first.
            right: the condition checked when the first holds.
        """
        self.left = left
        self.right = right
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when both conditions hold."""
        return self.left(hook) and self.right(hook)

    @property
    @override
    def operands(self) -> tuple[Condition, ...]:
        """The two conditions."""
        return self.left, self.right


class NotCondition(LogicalCondition, Generic[_C_co]):
    """Hold when the condition does not.

    Attributes:
        condition: the condition to negate.
    """

    condition: Final[_C_co]

    def __init__(self, condition: _C_co) -> None:
        """Initialize.

        Args:
            condition: the condition to negate.
        """
        self.condition = condition
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when the condition does not hold."""
        return not self.condition(hook)

    @property
    @override
    def operands(self) -> tuple[Condition, ...]:
        """The negated condition."""
        return (self.condition,)


class OrCondition(LogicalCondition, Generic[_L_co, _R_co]):
    """Hold when either condition holds.

    Attributes:
        left: the condition checked first.
        right: the condition checked when the first does not hold.
    """

    left: Final[_L_co]
    right: Final[_R_co]

    def __init__(self, left: _L_co, right: _R_co) -> None:
        """Initialize.

        Args:
            left: the condition checked first.
            right: the condition checked when the first does not hold.
        """
        self.left = left
        self.right = right
        return

    @override
    def __call__(self, hook: p.HookProtocol) -> bool:
        """Hold when either condition holds."""
        return self.left(hook) or self.right(hook)

    @property
    @override
    def operands(self) -> tuple[Condition, ...]:
        """The two conditions."""
        return self.left, self.right

"""Metric extraction, history and improvement checks for callbacks."""

from __future__ import annotations

import operator
import warnings

from collections.abc import Callable, Sequence
from typing import Any, Final, Generic, Literal, TypeVar

from drytorch.core import exceptions
from drytorch.core import protocols as p


__all__ = [
    'MetricExtractor',
    'MetricMonitor',
    'MetricTracker',
]


Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)
_Output_contra = TypeVar(
    '_Output_contra', bound=p.OutputType, contravariant=True
)
_Target_contra = TypeVar(
    '_Target_contra', bound=p.TargetType, contravariant=True
)

get_last: Final = operator.itemgetter(-1)


class MetricExtractor:
    """Handle extraction of metrics from trainer/validation protocols.

    This class is responsible for interfacing with trainer and validation
    protocols to extract metric values.

    Attributes:
        metric_spec: the metric specification (name or protocol instance).
        optional_source: the source computing the metric.
    """

    metric_spec: Final[p.ObjectiveProtocol[Any, Any] | str | None]
    optional_source: Final[p.SourceProtocol | None]
    _resolved_metric: p.ObjectiveProtocol[Any, Any] | None
    _resolved_metric_name: str | None

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Any, Any] | str | None = None,
        source: p.SourceProtocol | None = None,
    ) -> None:
        """Initialize.

        Args:
            metric: name of the metric to monitor or metric calculator instance.
            source: the source computing the metric.
        """
        self.metric_spec = metric
        self.optional_source = source
        self._resolved_metric = None
        self._resolved_metric_name = None
        return

    @property
    def metric_name(self) -> str | None:
        """Get the resolved metric name."""
        return self._resolved_metric_name

    def extract_metric_value(
        self,
        instance: p.TrainerProtocol[Input, Target, Output],
        tracker: MetricTracker[Output, Target],
    ) -> float:
        """Extract and return the metric value from the instance.

        Args:
            instance: trainer instance to extract from.
            tracker: MetricTracker to potentially update metric name.

        Returns:
            The extracted metric value.

        Raises:
            MetricNotFoundError: if the specified metric is not found.
        """
        source = self._get_source(instance)
        metric_name = self._resolved_metric_name
        if metric_name is None:
            metric_name, metric = self._resolve_metric(source)
            self._resolved_metric_name = metric_name
            self._resolved_metric = metric
            tracker.resolve_metric_name(metric_name)

        last_metrics = source.computed_metrics
        if metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(source.name, metric_name)

        return last_metrics[metric_name]

    def get_metric_best_is(self) -> Literal['higher', 'lower'] | None:
        """Get the best_is preference from the metric if available.

        Returns:
            ``'higher'`` or ``'lower'`` if the metric states its direction,
            ``None`` otherwise.
        """
        if self._resolved_metric is not None:
            return self._get_metric_best_is(self._resolved_metric)

        return self._get_metric_best_is(self.metric_spec)

    def _resolve_metric(
        self, source: p.SourceProtocol
    ) -> tuple[str, p.ObjectiveProtocol[Any, Any] | None]:
        if isinstance(self.metric_spec, p.ObjectiveProtocol):
            metric = self.metric_spec
        else:
            metric = source.objective

        metric_name = self._get_metric_name(metric)
        name = (
            self.metric_spec
            if isinstance(self.metric_spec, str)
            else metric_name
        )

        # the objective describes the monitored value only when names agree
        resolved_metric = metric if name == metric_name else None
        return name, resolved_metric

    def _get_source(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> p.SourceProtocol:
        if self.optional_source is None:
            if instance.validation is None:
                return instance

            return instance.validation

        return self.optional_source

    @staticmethod
    def _get_metric_name(
        metric: p.ObjectiveProtocol[Output, Target] | str,
    ) -> str:
        if isinstance(metric, str):
            return metric

        if isinstance(metric, p.NamedProtocol) and metric.name:
            return str(metric.name)

        return metric.__class__.__name__

    @staticmethod
    def _get_metric_best_is(
        metric: p.ObjectiveProtocol[Output, Target] | str | None,
    ) -> Literal['higher', 'lower'] | None:
        if isinstance(metric, p.HigherIsBetterProtocol):
            higher_is_better = metric.higher_is_better
        else:
            higher_is_better = None

        if higher_is_better is None:
            return None

        return 'higher' if higher_is_better else 'lower'


class MetricMonitor(Generic[Output, Target]):
    """Handle metric monitoring and alerts when performance stops increasing.

    Attributes:
        extractor: handles metric extraction from protocols.
        metric_tracker: handles metric value tracking and improvement detection.
    """

    extractor: Final[MetricExtractor]
    metric_tracker: Final[MetricTracker[Output, Target]]

    def __init__(
        self,
        metric: p.ObjectiveProtocol[Output, Target] | str | None = None,
        source: p.SourceProtocol | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """Initialize.

        Args:
            metric: name of the metric to monitor or metric calculator instance.
            source: the source computing the metric.
            min_delta: minimum absolute change to qualify as an improvement.
            patience: number of checks to wait before triggering callback.
            best_is: whether higher or lower metric values are better.
                Default 'auto' takes it from the metric when it states its
                direction, from initial measurements otherwise.
            filter_fn: function to aggregate recent metric values.
        """
        self.extractor = MetricExtractor(metric=metric, source=source)
        if best_is == 'auto':
            best_is = self.extractor.get_metric_best_is() or 'auto'

        initial_metric_name = None
        if isinstance(metric, str):
            initial_metric_name = metric

        self.metric_tracker = MetricTracker(
            metric_name=initial_metric_name,
            min_delta=min_delta,
            patience=patience,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        return

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
    def history(self) -> Sequence[float]:
        """Get the metric history."""
        return self.metric_tracker.history

    def is_better(self, value: float, reference: float) -> bool:
        """Check whether a value is better than a reference.

        Args:
            value: the value to compare.
            reference: the value to compare against.

        Returns:
            ``True`` if ``value`` is better than ``reference``.
        """
        return self.metric_tracker.is_better(value, reference)

    def is_improving(self) -> bool:
        """Determine if the model performance is improving.

        Returns:
            ``True`` if the performance is improving.

        Side Effects:
            Reduces the patience countdown when there is no improvement.
        """
        return self.metric_tracker.is_improving()

    def is_patient(self) -> bool:
        """Check whether to be patient.

        Returns:
            ``True`` while the patience has not run out.
        """
        return self.metric_tracker.is_patient()

    def record_metric_value(
        self, instance: p.TrainerProtocol[Any, Target, Output]
    ) -> None:
        """Register a new metric value from a monitored evaluation.

        Args:
            instance: trainer instance to extract metric from.

        Raises:
            MetricNotFoundError: if the specified metric is not found.
        """
        value = self.extractor.extract_metric_value(
            instance, self.metric_tracker
        )
        best_is = self.extractor.get_metric_best_is()
        if best_is is not None:
            self.metric_tracker.resolve_best_is(best_is)

        self.metric_tracker.add_value(value)
        return


class MetricTracker(Generic[_Output_contra, _Target_contra]):
    """Handle metric value tracking and improvement detection.

    This class is responsible for storing metric history, determining
    improvements, and managing patience countdown.

    Note: this class can be used to automatically modify the training strategy.
    Therefore, it does not follow the library conventions for a tracker.

    Attributes:
        best_is: whether higher or lower values are better, or ``'auto'`` to
            resolve it at runtime.
        min_delta: the minimum change required to qualify as an improvement.
        patience: number of checks to wait before triggering callback.
        filter_fn: function to aggregate recent metric values.
    """

    best_is: Final[Literal['auto', 'higher', 'lower']]
    filter_fn: Final[Callable[[Sequence[float]], float]]
    min_delta: Final[float]
    patience: Final[int]
    _metric_name: str | None
    _resolved_best_is: Literal['higher', 'lower'] | None
    _history: Final[list[float]]
    _patience_countdown: int
    _best_value: float | None

    def __init__(
        self,
        metric_name: str | None = None,
        min_delta: float = 1e-8,
        patience: int = 0,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
        filter_fn: Callable[[Sequence[float]], float] = get_last,
    ) -> None:
        """Initialize.

        Args:
            metric_name: name of the metric to track.
            min_delta: minimum absolute change to qualify as an improvement.
            patience: number of checks to wait before triggering callback.
            best_is: whether higher or lower metric values are better.
            filter_fn: function to aggregate recent metric values.
        """
        self._metric_name = metric_name
        self.best_is = best_is
        self._resolved_best_is = None if best_is == 'auto' else best_is
        self.filter_fn = filter_fn
        self.min_delta = min_delta
        self._validate_patience(patience)
        self.patience = patience
        self._history = []
        self._patience_countdown = patience
        self._best_value = None
        return

    @property
    def best_value(self) -> float:
        """Get the best result observed so far.

        Returns:
            the best filtered value (according to the 'best_is' criterion).

        Raises:
            ResultNotAvailableError: if no results have been logged yet.
        """
        if self._best_value is None:
            try:
                self._best_value = self._history[0]
            except IndexError as ie:
                raise exceptions.ResultNotAvailableError() from ie

        return self._best_value

    @property
    def filtered_value(self) -> float:
        """Get the current value.

        Returns:
            the current value aggregated from recent ones.

        Raises:
            ResultNotAvailableError: if no results have been logged yet.
        """
        if not self.history:
            raise exceptions.ResultNotAvailableError()

        return self.filter_fn(self.history)

    @property
    def history(self) -> Sequence[float]:
        """The recorded metric values."""
        return self._history

    @property
    def metric_name(self) -> str | None:
        """The name of the monitored metric, once known."""
        return self._metric_name

    @property
    def resolved_best_is(self) -> Literal['higher', 'lower']:
        """The direction in use: ``best_is``, or the one resolved at runtime.

        Raises:
            DirectionNotResolvedError: if the direction is not known yet.
        """
        if self._resolved_best_is is None:
            raise exceptions.DirectionNotResolvedError()

        return self._resolved_best_is

    def add_value(self, value: float) -> None:
        """Add a new metric value to the history.

        Args:
            value: the metric value to add.
        """
        self._history.append(value)
        return

    def is_better(self, value: float, reference: float) -> bool:
        """Determine if the value is better than a reference value.

        When best_is is in 'auto' mode, it is assumed that the given value is
        better than the first recorded one.

        Args:
            value: the value to compare.
            reference: the reference.

        Returns:
            True if value is a potential improvement, False otherwise.

        Side Effects:
            An unresolved direction is guessed from the first two values,
            emitting :class:`~drytorch.core.exceptions.DirectionGuessedWarning`.
        """
        if value != value:  # check for NaN
            return False

        if self._resolved_best_is is None:
            if len(self.history) < 2:
                return True

            if self.history[0] > self.history[1]:
                self._resolved_best_is = 'lower'
            else:
                self._resolved_best_is = 'higher'

            warnings.warn(
                exceptions.DirectionGuessedWarning(
                    self._resolved_best_is, self.metric_name
                ),
                stacklevel=2,
            )

        if self._resolved_best_is == 'lower':
            return reference - self.min_delta > value

        return reference + self.min_delta < value

    def is_improving(self) -> bool:
        """Determine if the model performance is improving.

        Returns:
            True if there has been an improvement, False otherwise.

        Side Effects:
            If there is no improvement, the patience countdown is reduced.
            Otherwise, it is restored to the maximum.
        """
        if len(self.history) <= 1:
            return True

        aggregated_value = self.filtered_value

        if self.is_better(aggregated_value, self.best_value):
            self._best_value = aggregated_value
            self._patience_countdown = self.patience
            return True

        self._patience_countdown -= 1
        return False

    def is_patient(self) -> bool:
        """Check whether to be patient.

        Returns:
            ``True`` while the patience has not run out.
        """
        return self._patience_countdown > 0

    def reset_patience(self) -> None:
        """Reset patience countdown to the maximum."""
        self._patience_countdown = self.patience
        return

    def resolve_best_is(self, best_is: Literal['higher', 'lower']) -> None:
        """Set the direction if it is still automatic.

        Args:
            best_is: whether higher or lower values are better.
        """
        if self._resolved_best_is is None:
            self._resolved_best_is = best_is

        return

    def resolve_metric_name(self, metric_name: str) -> None:
        """Set the metric name if it is not known yet.

        Args:
            metric_name: the name of the monitored metric.
        """
        if self._metric_name is None:
            self._metric_name = metric_name

        return

    @staticmethod
    def _validate_patience(patience: int) -> None:
        if patience < 0:
            raise ValueError('Patience must be a non-negative integer.')

        return

"""Module containing classes to create and combine loss and metrics.

The interface is similar to https://github.com/Lightning-AI/torchmetrics,
with stricter typing and simpler construction. MetricCollection and
CompositionalMetric from torchmetrics change their state; here a functional
approach is preferred.
"""

from __future__ import annotations

import abc
import copy
import functools
import operator
import warnings

from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    Self,
    TypeVar,
    overload,
)

import torch

from typing_extensions import override

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import aggregators


__all__ = [
    'AverageObjective',
    'DictLoss',
    'JoinedLoss',
    'JoinedMetrics',
    'Loss',
    'LossWithMetrics',
    'Metric',
    'MetricCollection',
    'MetricTracker',
    'Objective',
    'compute_metrics',
    'join',
]


Output = TypeVar('Output', bound=p.OutputType)
Target = TypeVar('Target', bound=p.TargetType)
_Output_contra = TypeVar(
    '_Output_contra', bound=p.OutputType, contravariant=True
)
_Target_contra = TypeVar(
    '_Target_contra', bound=p.TargetType, contravariant=True
)
Tensor = torch.Tensor

# precedence of the top-level operation in a loss formula
_SUM: Final = 0
_PRODUCT: Final = 1
_NEGATION: Final = 2
_POWER: Final = 3
_TERM: Final = 4
_OPERATORS: Final[dict[str, tuple[int, Callable[[Tensor, Tensor], Tensor]]]] = {
    '+': (_SUM, operator.add),
    '-': (_SUM, operator.sub),
    'x': (_PRODUCT, operator.mul),
    '/': (_PRODUCT, operator.truediv),
}


class Objective(p.ObjectiveProtocol[_Output_contra, _Target_contra], abc.ABC):
    """Abstract base class for metrics or losses."""

    def __init__(self) -> None:
        """Initialize."""
        self._aggregator = aggregators.Aggregator[
            aggregators.AbstractAccumulator[Tensor, Tensor]
        ]()
        self._cached_values: dict[str, Tensor] = {}
        return

    @override
    def compute(self) -> dict[str, Tensor]:
        """Return the aggregated objective value(s).

        Despite the name, which follows common practice, this method caches
        previous computed values and returns them if available.

        Returns:
            A dictionary of computed metric values.
        """
        if not self._aggregator:
            warnings.warn(
                exceptions.ComputedBeforeUpdatedWarning(self), stacklevel=1
            )

        if not self._cached_values:
            self._cached_values = self._compute()

        return dict(self._cached_values)

    @override
    def update(
        self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, Tensor]:
        """Updates the objective's internal state with new outputs and targets.

        Args:
            outputs: the model outputs.
            targets: the ground truth targets.

        Returns:
            A dictionary of the calculated metric values for the current update.
        """
        results = self.calculate(outputs, targets)
        accumulator_cls = self._get_accumulator_cls()
        self._aggregator += {
            key: accumulator_cls.from_value(value)
            for key, value in results.items()
        }
        self._cached_values.clear()
        return results

    @override
    def reset(self) -> None:
        """Resets the internal state of the instance."""
        self._aggregator.clear()
        self._cached_values.clear()
        return

    @abc.abstractmethod
    def calculate(
        self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, Tensor]:
        """Method responsible for the calculations.

        Args:
            outputs: model outputs.
            targets: ground truth.

        Returns:
            A dictionary of calculated metric values.
        """

    def copy(self) -> Self:
        """Create a (deep)copy of self.

        Returns:
            A deep copy of the objective.
        """
        return copy.deepcopy(self, {})

    def merge_state(self, other: Objective[Any, Any]) -> None:
        """Merge metric states.

        Args:
            other: metric to be merged with.

        Raises:
            AccumulatorMismatchError: if the two objectives use different
                accumulators.
        """
        accumulator_cls = self._get_accumulator_cls()
        other_accumulator_cls = other._get_accumulator_cls()
        if other_accumulator_cls is not accumulator_cls:
            raise exceptions.AccumulatorMismatchError(
                accumulator_cls, other_accumulator_cls
            )

        self._aggregator += other._aggregator.accumulators
        self._cached_values.clear()
        return

    def sync(self) -> None:
        """Synchronize metric states across processes."""
        self._aggregator.sync()
        self._cached_values.clear()
        return

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deep copy magic method.

        Args:
            memo: dictionary of already copied objects.

        Returns:
            A deep copy of the object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    def _compute(self) -> dict[str, Tensor]:
        """Computes the objective value(s)."""
        return {
            key: acc.reduce()
            for key, acc in self._aggregator.accumulators.items()
        }

    @classmethod
    @abc.abstractmethod
    def _get_accumulator_cls(
        cls,
    ) -> type[aggregators.AbstractAccumulator[Tensor, Tensor]]:
        """Returns the accumulator class."""


class AverageObjective(Objective[_Output_contra, _Target_contra], abc.ABC):
    """Class defining the default aggregation."""

    @classmethod
    @override
    def _get_accumulator_cls(
        cls,
    ) -> type[aggregators.AbstractAccumulator[Tensor, Tensor]]:
        return aggregators.TorchMeanAccumulator


class MetricCollection(AverageObjective[Output, Target]):
    """A collection of multiple metrics.

    Because :class:`AverageObjective` aggregates values across batches,
    the provided functions should ideally return unreduced ``(B,)`` tensors
    (where ``B`` is the batch size) rather than pre-reduced scalars. This
    ensures the final average correctly weights each batch by its size.

    Attributes:
        fn: functions returning multiple named metric values.
        named_fn: dictionary of named functions returning a single value.
    """

    fn: Final[tuple[Callable[[Output, Target], dict[str, Tensor]], ...]]
    named_fn: Final[Mapping[str, Callable[[Output, Target], Tensor]]]

    def __init__(
        self,
        *fn: Callable[[Output, Target], dict[str, Tensor]],
        **named_fn: Callable[[Output, Target], Tensor],
    ) -> None:
        """Initialize.

        Args:
            *fn: functions returning multiple named metric values.
            **named_fn: dictionary of named functions returning a single value.
        """
        super().__init__()
        self.fn = tuple(fn)
        self.named_fn = named_fn
        return

    @override
    def calculate(self, outputs: Output, targets: Target) -> dict[str, Tensor]:
        """Calculates the values for all metrics in the collection.

        Args:
            outputs: the model outputs.
            targets: the ground truth targets.

        Returns:
            A dictionary of calculated metric values.
        """
        result = {
            name: fn(outputs, targets) for name, fn in self.named_fn.items()
        }
        for f in self.fn:
            result |= f(outputs, targets)

        return result

    def __or__(
        self, other: MetricCollection[Output, Target]
    ) -> MetricCollection[Output, Target]:
        """Constructor using existing MetricCollection objects as templates.

        This class does not aggregate the states. If you intend to do this,
        use the merge_state method separately.

        Args:
            other: another MetricCollection object to combine with.

        Returns:
            A new instance containing metrics from both instances.

        Raises:
            RepeatedMetricsError: if a name maps to different functions.
        """
        _check_distinct_names(self.named_fn, other.named_fn)
        return MetricCollection(
            *self.fn, *other.fn, **{**self.named_fn, **other.named_fn}
        )


class Metric(MetricCollection[Output, Target]):
    """A single named metric.

    Attributes:
        name: identifier for the metric.
        higher_is_better: ``True`` if higher values indicate better performance.
    """

    name: Final[str]
    higher_is_better: Final[bool | None]

    def __init__(
        self,
        fn: Callable[[Output, Target], Tensor],
        /,
        name: str,
        higher_is_better: bool | None = None,
    ) -> None:
        """Initialize.

        Args:
            fn: the callable that computes the metric value.
            name: identifier for the metric.
            higher_is_better: ``True`` if higher values indicate better
                performance, ``False`` if lower values are better, ``None`` if
                unspecified.
        """
        super().__init__(**{name: fn})
        self.name = name
        self.higher_is_better = higher_is_better
        return


class LossWithMetrics(
    MetricCollection[Output, Target],
    p.LossProtocol[Output, Target],
):
    """Collection of metrics, one of which serves as a loss.

    Attributes:
        name: identifier for the loss.
        higher_is_better: ``True`` if higher values indicate better performance.
        formula: string representation of the loss formula.
        criterion: logic extracting a loss value from computed value.
    """

    name: Final[str]
    higher_is_better: Final[bool]
    formula: Final[str]
    criterion: Final[Callable[[dict[str, Tensor]], Tensor]]

    def __init__(
        self,
        *fn: Callable[[Output, Target], dict[str, Tensor]],
        criterion: Callable[[dict[str, Tensor]], Tensor],
        name: str = 'Loss',
        higher_is_better: bool = False,
        formula: str = '',
        **named_fn: Callable[[Output, Target], Tensor],
    ) -> None:
        """Initialize.

        Args:
            *fn: functions returning multiple named metric values.
            criterion: logic extracting a loss value from computed value.
            name: identifier for the loss.
            higher_is_better: ``True`` if higher values indicate better
                performance. Defaults to ``False``.
            formula: string representation of the loss formula. Default is
                the name in brackets.
            **named_fn: dictionary of named functions returning a single value.
        """
        self.name = name
        self.higher_is_better = higher_is_better
        self.formula = formula or f'[{name}]'
        super().__init__(*fn, **named_fn)
        self.criterion = criterion
        return

    @override
    def calculate(self, outputs: Output, targets: Target) -> dict[str, Tensor]:
        """Calculates the loss and all associated metric values.

        Args:
            outputs: the model outputs.
            targets: the ground truth targets.

        Returns:
            A dictionary containing the calculated loss and metric values.
        """
        all_metrics = super().calculate(outputs, targets)
        loss_value = self.criterion(all_metrics)
        all_metrics.pop(self.name, None)  # the loss value wins a name clash
        return {self.name: loss_value} | all_metrics

    @override
    def forward(self, outputs: Output, targets: Target) -> Tensor:
        """Performs a forward pass, updates metrics, and computes the loss.

        Args:
            outputs: the model outputs.
            targets: the ground truth targets.

        Returns:
            The computed loss value.
        """
        metrics = self.update(outputs, targets)
        return self.criterion(metrics).mean()

    def __neg__(self) -> LossWithMetrics[Output, Target]:
        """Return the negated loss.

        Returns:
            The negated loss, with the opposite direction.
        """
        criterion = self.criterion
        return LossWithMetrics(
            *self.fn,
            criterion=lambda x: -criterion(x),
            name='Negative ' + self.name,
            higher_is_better=not self.higher_is_better,
            formula='-' + _parenthesize(self.formula, _POWER),
            **self.named_fn,
        )

    def __add__(
        self,
        other: LossWithMetrics[Output, Target] | float,
    ) -> LossWithMetrics[Any, Any]:
        """Return the sum with another loss or a number.

        Args:
            other: the loss or number to add.

        Returns:
            The sum, or this loss if ``other`` is zero.

        Raises:
            ValueError: if ``other`` is a loss with the opposite direction.
            RepeatedMetricsError: if a metric name maps to different functions.
        """
        if other == 0:
            return self

        return self._combine(other, '+')

    def __radd__(self, other: float) -> LossWithMetrics[Any, Any]:
        """Return a number plus the loss.

        Args:
            other: the number to add.

        Returns:
            The sum, or this loss if ``other`` is zero.
        """
        if other == 0:
            return self

        return self._combine(other, '+', reflected=True)

    def __sub__(
        self,
        other: LossWithMetrics[Output, Target] | float,
    ) -> LossWithMetrics[Output, Target]:
        """Return the difference with another loss or a number.

        Args:
            other: the loss or number to subtract.

        Returns:
            The difference.

        Raises:
            ValueError: if ``other`` is a loss with the same direction.
            RepeatedMetricsError: if a metric name maps to different functions.
        """
        return self._combine(other, '-')

    def __rsub__(self, other: float) -> LossWithMetrics[Output, Target]:
        """Return a number minus the loss.

        Args:
            other: the number from which to subtract the loss.

        Returns:
            The difference, with the opposite direction.
        """
        return self._combine(other, '-', reflected=True)

    def __mul__(
        self,
        other: LossWithMetrics[Output, Target] | float,
    ) -> LossWithMetrics[Output, Target]:
        """Return the product with another loss or a number.

        Args:
            other: the loss or number to multiply by.

        Returns:
            The product, or this loss if ``other`` is one.

        Raises:
            ValueError: if ``other`` is a loss with the opposite direction.
            RepeatedMetricsError: if a metric name maps to different functions.
        """
        if other == 1:
            return self

        return self._combine(other, 'x')

    def __rmul__(self, other: float) -> LossWithMetrics[Output, Target]:
        """Return a number times the loss.

        Args:
            other: the number to multiply by.

        Returns:
            The product, or this loss if ``other`` is one.
        """
        if other == 1:
            return self

        return self._combine(other, 'x', reflected=True)

    def __truediv__(
        self,
        other: LossWithMetrics[Output, Target] | float,
    ) -> LossWithMetrics[Output, Target]:
        """Return the quotient by another loss or a number.

        Args:
            other: the loss or number to divide by.

        Returns:
            The quotient, or this loss if ``other`` is one.

        Raises:
            ValueError: if ``other`` is a loss with the same direction.
            RepeatedMetricsError: if a metric name maps to different functions.
        """
        if other == 1:
            return self

        return self._combine(other, '/')

    def __rtruediv__(self, other: float) -> LossWithMetrics[Output, Target]:
        """Return a number divided by the loss.

        Args:
            other: the number to divide.

        Returns:
            The quotient.
        """
        return self._combine(other, '/', reflected=True)

    def __pow__(self, other: float) -> LossWithMetrics[Output, Target]:
        """Return the loss raised to a power.

        A negative exponent flips the direction. This assumes positive values:
        for a loss that takes negative values, such as ``(-loss) ** 2``, the
        direction can be wrong.

        Args:
            other: the exponent.

        Returns:
            The power, or this loss if ``other`` is one.
        """
        if other == 1:
            return self

        criterion = self.criterion

        def _power(x: dict[str, Tensor]) -> Tensor:
            value = criterion(x)
            if not torch.is_floating_point(value):
                value = value.float()

            return value**other

        return LossWithMetrics(
            *self.fn,
            criterion=_power,
            name='Loss',
            higher_is_better=self.higher_is_better != (other < 0),
            formula=f'{_parenthesize(self.formula, _TERM)}^{other}',
            **self.named_fn,
        )

    def __repr__(self):
        """Return the class name with the formula."""
        return f'{self.__class__.__name__}({self.formula})'

    def _combine(
        self,
        other: LossWithMetrics[Output, Target] | float,
        symbol: Literal['+', '-', 'x', '/'],
        reflected: bool = False,
    ) -> LossWithMetrics[Output, Target]:
        precedence, operation = _OPERATORS[symbol]
        inverse = symbol in ('-', '/')
        if isinstance(other, LossWithMetrics):
            if (self.higher_is_better == other.higher_is_better) == inverse:
                required = 'opposite' if inverse else 'the same'
                msg = (
                    f'{self!r} {symbol} {other!r} needs {required} directions.'
                )
                raise ValueError(msg)

            _check_distinct_names(self.named_fn, other.named_fn)
            fn = self.fn + other.fn
            named_fn = {**self.named_fn, **other.named_fn}
            other_formula = other.formula
            other_criterion = other.criterion
            higher_is_better = self.higher_is_better
        elif isinstance(other, float | int):
            fn = self.fn
            named_fn = self.named_fn
            other_formula = str(other)
            value = torch.tensor(other)

            def _constant(_: dict[str, Tensor]) -> Tensor:
                return value

            other_criterion = _constant
            flips = {
                '+': False,
                '-': reflected,
                'x': other < 0,
                '/': other > 0 if reflected else other < 0,
            }
            higher_is_better = self.higher_is_better != flips[symbol]
        else:
            raise TypeError(f'Unsupported type for operation: {type(other)}')

        first_criterion, second_criterion = self.criterion, other_criterion
        first, second = self.formula, other_formula
        if reflected:
            first_criterion, second_criterion = (
                second_criterion,
                first_criterion,
            )
            first, second = second, first

        def _combined(x: dict[str, Tensor]) -> Tensor:
            return operation(first_criterion(x), second_criterion(x))

        # the right operand of - and / needs parentheses at equal precedence
        second_precedence = precedence + 1 if inverse else precedence
        formula = (
            f'{_parenthesize(first, precedence)} {symbol} '
            f'{_parenthesize(second, second_precedence)}'
        )
        return LossWithMetrics(
            *fn,
            criterion=_combined,
            name='Combined Loss',
            higher_is_better=higher_is_better,
            formula=formula,
            **named_fn,
        )


class Loss(LossWithMetrics[Output, Target]):
    """Subclass for simple losses with a convenient constructor."""

    def __init__(
        self,
        fn: Callable[[Output, Target], Tensor],
        /,
        name: str,
        higher_is_better: bool = False,
    ):
        """Initialize.

        Args:
            fn: the callable to calculate the loss.
            name: the name for the loss.
            higher_is_better: ``True`` if higher values indicate better
                performance. Defaults to ``False``.
        """
        super().__init__(
            criterion=operator.itemgetter(name),
            name=name,
            higher_is_better=higher_is_better,
            formula=f'[{name}]',
            **{name: fn},
        )
        return


class DictLoss(LossWithMetrics[Output, Target]):
    """Subclass for losses whose function also returns auxiliary metrics."""

    def __init__(
        self,
        fn: Callable[[Output, Target], dict[str, Tensor]],
        /,
        name: str,
        higher_is_better: bool = False,
    ):
        """Initialize.

        Args:
            fn: callable returning named metric values, including the loss.
            name: key to extract as the loss and identifier for the loss.
            higher_is_better: ``True`` if higher values indicate better
                performance. Defaults to ``False``.
        """
        super().__init__(
            fn,
            criterion=operator.itemgetter(name),
            name=name,
            higher_is_better=higher_is_better,
            formula=f'[{name}]',
        )
        return


class JoinedMetrics(p.ObjectiveProtocol[_Output_contra, _Target_contra]):
    """Objective joining objectives of any kind.

    Attributes:
        objectives: the joined objectives.
    """

    objectives: Final[
        tuple[p.ObjectiveProtocol[_Output_contra, _Target_contra], ...]
    ]

    def __init__(
        self, *objectives: p.ObjectiveProtocol[_Output_contra, _Target_contra]
    ) -> None:
        """Initialize.

        Args:
            *objectives: the objectives to join.
        """
        super().__init__()
        self.objectives = objectives
        return

    @override
    def compute(self) -> dict[str, Tensor]:
        """Return the computed values of all the objectives.

        Returns:
            The computed values by name.

        Raises:
            RepeatedMetricsError: if two objectives compute the same names.
            ComputedMetricsTypeError: if an objective computes nothing.
        """
        computed: dict[str, Tensor] = {}
        for objective in self.objectives:
            values = _named_values(objective)
            shared = computed.keys() & values.keys()
            if shared:
                raise exceptions.RepeatedMetricsError(sorted(shared))

            computed.update(values)

        return computed

    @override
    def reset(self) -> None:
        """Reset the state of all the objectives."""
        for objective in self.objectives:
            objective.reset()

        return

    def sync(self) -> None:
        """Synchronize the objectives that support it across processes."""
        for objective in self.objectives:
            if isinstance(objective, p.SyncProtocol):
                objective.sync()
            else:
                issue = f'{objective.__class__.__name__} not synchronized'
                recommend = 'implement a `sync` method'
                warnings.warn(
                    exceptions.ObjectiveSyncWarning(issue, recommend),
                    stacklevel=2,
                )

        return

    @override
    def update(self, outputs: _Output_contra, targets: _Target_contra) -> None:
        """Update all the objectives with new outputs and targets.

        Args:
            outputs: the model outputs.
            targets: the ground truth targets.
        """
        for objective in self.objectives:
            objective.update(outputs, targets)

        return


class JoinedLoss(
    JoinedMetrics[_Output_contra, _Target_contra],
    p.LossProtocol[_Output_contra, _Target_contra],
):
    """Loss joining a loss with objectives of any kind.

    Attributes:
        objectives: the joined objectives, the loss first.
        loss: the loss to optimize.
        name: the name of the loss, or of its class if it has none.
    """

    loss: Final[p.LossProtocol[_Output_contra, _Target_contra]]
    name: Final[str]

    def __init__(
        self,
        loss: p.LossProtocol[_Output_contra, _Target_contra],
        /,
        *objectives: p.ObjectiveProtocol[_Output_contra, _Target_contra],
    ) -> None:
        """Initialize.

        Args:
            loss: the loss to optimize.
            *objectives: the objectives to track alongside the loss.
        """
        super().__init__(loss, *objectives)
        self.loss = loss
        self.name = getattr(loss, 'name', loss.__class__.__name__)
        return

    @override
    def forward(
        self, outputs: _Output_contra, targets: _Target_contra, /
    ) -> Tensor:
        """Compute the loss and update the other objectives.

        Args:
            outputs: the model outputs.
            targets: the ground truth targets.

        Returns:
            The computed loss.
        """
        for objective in self.objectives[1:]:
            objective.update(outputs, targets)

        return self.loss.forward(outputs, targets)


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
        filter_fn: Callable[[Sequence[float]], float] = operator.itemgetter(-1),
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

            msg = 'Monitoring {} assuming {} is best.'
            msg = msg.format(self.metric_name, self._resolved_best_is)
            warnings.warn(msg, stacklevel=2)

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


def check_device(
    calculator: p.ObjectiveProtocol[Any, Any], device: torch.device
) -> None:
    """Check the metrics returned by the calculator are on the given device.

    Args:
        calculator: an objective to check.
        device: the device to check against.

    Raises:
        DeviceMismatchError: if a metric is on a different device.
        ComputedMetricsTypeError: if the objective computes nothing.
    """
    for name, value in _named_values(calculator).items():
        if value.device.type != device.type:
            raise exceptions.DeviceMismatchError(name, value.device, device)

    return


def compute_metrics(
    calculator: p.ObjectiveProtocol[Any, Any],
) -> Mapping[str, float]:
    """Compute and represent the metrics as a mapping of named values.

    Args:
        calculator: an objective from which to compute metrics.

    Returns:
        A mapping of metric names to their float values.

    Raises:
        ComputedMetricsTypeError: if the objective computes nothing.
    """
    named_values = _named_values(calculator)
    return {name: value.item() for name, value in named_values.items()}


@overload
def join(
    objective: LossWithMetrics[Output, Target],
    /,
    *metrics: MetricCollection[Output, Target],
) -> LossWithMetrics[Output, Target]: ...


@overload
def join(
    objective: MetricCollection[Output, Target],
    /,
    *metrics: MetricCollection[Output, Target],
) -> MetricCollection[Output, Target]: ...


@overload
def join(
    objective: p.LossProtocol[Output, Target],
    /,
    *metrics: p.ObjectiveProtocol[Output, Target],
) -> p.LossProtocol[Output, Target]: ...


@overload
def join(
    objective: p.ObjectiveProtocol[Output, Target],
    /,
    *metrics: p.ObjectiveProtocol[Output, Target],
) -> p.ObjectiveProtocol[Output, Target]: ...


def join(
    objective: p.ObjectiveProtocol[Output, Target],
    /,
    *metrics: p.ObjectiveProtocol[Output, Target],
) -> p.ObjectiveProtocol[Output, Target]:
    """Join objectives into one of the same kind as the first.

    Args:
        objective: the first objective.
        *metrics: the objectives to join as metrics.

    Returns:
        The joined objective.

    Raises:
        RepeatedMetricsError: if metric collections compute the same names.
    """
    all_collections = all(isinstance(obj, MetricCollection) for obj in metrics)
    if isinstance(objective, LossWithMetrics) and all_collections:
        merged: MetricCollection[Output, Target] = functools.reduce(
            operator.or_, metrics, objective
        )
        return LossWithMetrics(
            *merged.fn,
            criterion=objective.criterion,
            name=objective.name,
            higher_is_better=objective.higher_is_better,
            formula=objective.formula,
            **merged.named_fn,
        )

    if isinstance(objective, MetricCollection) and all_collections:
        return functools.reduce(operator.or_, metrics, objective)

    if isinstance(objective, p.LossProtocol):
        return JoinedLoss(objective, *metrics)

    return JoinedMetrics(objective, *metrics)


def _check_distinct_names(
    first: Mapping[str, Callable[..., Tensor]],
    second: Mapping[str, Callable[..., Tensor]],
) -> None:
    shared = {
        name
        for name in first.keys() & second.keys()
        if first[name] is not second[name]
    }
    if shared:
        raise exceptions.RepeatedMetricsError(sorted(shared))

    return


def _get_precedence(formula: str) -> int:
    # the weakest operation outside brackets and parentheses
    depth = 0
    top_level = ''
    for char in formula:
        if char in '([':
            depth += 1
        elif char in ')]':
            depth -= 1
        elif not depth:
            top_level += char

    if ' + ' in top_level or ' - ' in top_level:
        return _SUM

    if ' x ' in top_level or ' / ' in top_level:
        return _PRODUCT

    if top_level.startswith('-'):
        return _NEGATION

    if '^' in top_level:
        return _POWER

    return _TERM


def _named_values(
    objective: p.ObjectiveProtocol[Any, Any],
) -> Mapping[str, Tensor]:
    computed = objective.compute()
    if isinstance(computed, Mapping):
        return computed

    if isinstance(computed, Tensor):
        return {objective.__class__.__name__: computed}

    raise exceptions.ComputedMetricsTypeError(type(computed))


def _parenthesize(formula: str, precedence: int) -> str:
    if _get_precedence(formula) < precedence:
        return f'({formula})'

    return formula

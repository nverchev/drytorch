"""Module containing classes to create and combine loss and metrics.

The interface is similar to https://github.com/Lightning-AI/torchmetrics,
with stricter typing and simpler construction. MetricCollection and
CompositionalMetric from torchmetrics change their state; here a functional
approach is preferred.
"""

from __future__ import annotations

import abc
import copy
import operator
import warnings

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Literal, Self, TypeVar

import torch

from typing_extensions import override

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.utils import statistics


_Output_contra = TypeVar(
    '_Output_contra', bound=p.OutputType, contravariant=True
)
_Target_contra = TypeVar(
    '_Target_contra', bound=p.TargetType, contravariant=True
)

_Tensor = torch.Tensor


class Objective(
    p.ObjectiveProtocol[_Output_contra, _Target_contra], metaclass=abc.ABCMeta
):
    """Abstract base class for metrics or losses."""

    def __init__(
        self,
        **named_metric_fun: Callable[
            [_Output_contra, _Target_contra],
            _Tensor,
        ],
    ) -> None:
        """Initializes the Objective with a dictionary of metric functions.

        Args:
            **named_metric_fun: named functions that compute the metric.
        """
        self._aggregator = statistics.TorchAverager()
        self.named_metric_fun = named_metric_fun

    @override
    def compute(self: Self) -> dict[str, _Tensor]:
        """Return the aggregated objective value(s).

        Despite the name, which follows common practice, this method caches
        previous computed values and returns them if available.

        Returns:
            A dictionary of computed metric values.
        """
        if not self._aggregator:
            # noinspection PyArgumentEqualDefault
            warnings.warn(
                exceptions.ComputedBeforeUpdatedWarning(self), stacklevel=1
            )

        return self._aggregator.reduce()

    @override
    def update(
        self: Self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, _Tensor]:
        """Updates the objective's internal state with new outputs and targets.

        Args:
            outputs: The model outputs.
            targets: The ground truth targets.

        Returns:
            A dictionary of the calculated metric values for the current update.
        """
        results = self.calculate(outputs, targets)
        self._aggregator += {
            key: value.detach() for key, value in results.items()
        }
        return results

    @override
    def reset(self: Self) -> None:
        """Resets the internal state of the instance."""
        self._aggregator.clear()
        return

    @abc.abstractmethod
    def calculate(
        self: Self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, _Tensor]:
        """Method responsible for the calculations.

        Args:
            outputs: model outputs.
            targets: ground truth.
        """

    def copy(self) -> Self:
        """Create a (deep)copy of self."""
        return copy.deepcopy(self, {})

    def merge_state(self: Self, other: Self) -> None:
        """Merge metric states.

        Args:
            other: metric to be merged with.
        """
        # pylint: disable=protected-access
        self._aggregator += other._aggregator
        return

    def __or__(
        self, other: Objective[_Output_contra, _Target_contra]
    ) -> MetricCollection[_Output_contra, _Target_contra]:
        """Combines two Objectives into a MetricCollection.

        Args:
            other: The other Objective to combine with.

        Returns:
            A new MetricCollection containing metrics from both instances.
        """
        named_metric_fun = self.named_metric_fun | other.named_metric_fun
        return MetricCollection(**named_metric_fun)

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deep copy magic method.

        Args:
            memo: Dictionary of already copied objects.

        Returns:
            A deep copy of the object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            result.__dict__[k] = copy.deepcopy(v, memo)

        return result


class MetricCollection(Objective[_Output_contra, _Target_contra]):
    """A collection of multiple metrics."""

    @override
    def calculate(
        self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, _Tensor]:
        """Calculates the values for all metrics in the collection.

        Args:
            outputs: The model outputs.
            targets: The ground truth targets.

        Returns:
            A dictionary of calculated metric values.
        """
        return dict_apply(self.named_metric_fun, outputs, targets)


class Metric(Objective[_Output_contra, _Target_contra]):
    """A single metric."""

    def __init__(
        self,
        fun: Callable[[_Output_contra, _Target_contra], _Tensor],
        /,
        *,
        name: str,
        higher_is_better: bool | None = None,
    ) -> None:
        """Constructor.

        Args:
            fun: callable that computes the metric value.
            name: identifier for the metric.
            higher_is_better: True if higher values indicate better performance,
                False if lower values are better, None if unspecified.
        """
        super().__init__(**{name: fun})
        self.fun = fun
        self.name = name
        self.higher_is_better = higher_is_better

    @override
    def calculate(
        self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, _Tensor]:
        """Calculates the value of this single metric.

        Args:
            outputs: The model outputs.
            targets: The ground truth targets.

        Returns:
            A dictionary containing the calculated metric value.
        """
        return {self.name: self.fun(outputs, targets)}


class LossBase(
    MetricCollection[_Output_contra, _Target_contra],
    p.LossProtocol[_Output_contra, _Target_contra],
    metaclass=abc.ABCMeta,
):
    """Collection of metrics, one of which serves as a loss."""

    def __init__(
        self,
        criterion: Callable[[dict[str, _Tensor]], _Tensor],
        name: str,
        higher_is_better: bool = False,
        formula: str = 'Loss',
        **named_fun: Callable[[_Output_contra, _Target_contra], _Tensor],
    ) -> None:
        """Constructor.

        Args:
            criterion: function extracting a loss value from metric functions.
            name: identifier for the loss.
            higher_is_better: True if higher values indicate better performance,
                False if lower values are better.
            formula: string representation of the loss formula.
            **named_fun: dictionary of named metric functions.
        """
        self.name = name
        self.higher_is_better = higher_is_better
        self.formula = formula
        super().__init__(**named_fun)
        self.criterion = criterion
        return

    @override
    def forward(
        self, outputs: _Output_contra, targets: _Target_contra
    ) -> _Tensor:
        """Performs a forward pass, updates metrics, and computes the loss.

        Args:
            outputs: The model outputs.
            targets: The ground truth targets.

        Returns:
            The computed loss value.
        """
        metrics = self.update(outputs, targets)
        return self.criterion(metrics).mean()

    def __or__(
        self, other: Objective[_Output_contra, _Target_contra]
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Combines a LossBase with another Objective using the OR operator.

        Args:
            other: The other Objective to combine with.

        Returns:
            A new CompositionalLoss containing metrics from both instances.
        """
        named_metric_fun = self.named_metric_fun | other.named_metric_fun
        return CompositionalLoss(
            criterion=self.criterion,
            name=self.name,
            higher_is_better=self.higher_is_better,
            formula=self.formula,
            **named_metric_fun,
        )

    def _combine(
        self,
        other: LossBase[_Output_contra, _Target_contra] | float,
        operation: Callable[[_Tensor, _Tensor], _Tensor],
        op_fmt: str,
        requires_parentheses: bool = True,
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Support operations between losses or a loss and a float.

        Args:
            other: The other loss or float to combine with.
            operation: The callable operation to apply (e.g., operator.add).
            op_fmt: The format string for the combined formula.
            requires_parentheses: Whether to wrap sub-formulas in parentheses.

        Returns:
            A new CompositionalLoss representing the combined loss.
        """
        if isinstance(other, LossBase):
            named_metric_fun = self.named_metric_fun | other.named_metric_fun
            str_first = self.formula
            str_second = other.formula

            # apply should combine losses that share the same direction
            self._check_same_direction(other)

            def _combined(x: dict[str, _Tensor]) -> _Tensor:
                return operation(self.criterion(x), other.criterion(x))

        elif isinstance(other, float | int):
            named_metric_fun = self.named_metric_fun
            str_first = str(other)
            str_second = self.formula

            def _combined(x: dict[str, _Tensor]) -> _Tensor:
                return operation(self.criterion(x), torch.tensor(other))

        else:
            raise TypeError(f'Unsupported type for operation: {type(other)}')

        if not requires_parentheses:
            str_first = self._remove_outer_parentheses(str_first)
            str_second = self._remove_outer_parentheses(str_second)

        formula = op_fmt.format(str_first, str_second)
        return CompositionalLoss(
            criterion=_combined,
            higher_is_better=self.higher_is_better,
            formula=formula,
            **named_metric_fun,
        )

    def __neg__(self) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Negates the loss.

        Returns:
            A new CompositionalLoss representing the negated loss.
        """
        return CompositionalLoss(
            criterion=lambda x: -self.criterion(x),
            higher_is_better=not self.higher_is_better,
            formula=f'-{self.formula}',
            **self.named_metric_fun,
        )

    def __add__(
        self,
        other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss[Any, Any]:
        """Adds another loss or a float to this loss.

        Args:
            other: The other loss or float to add.

        Returns:
            A new CompositionalLoss representing the sum.
        """
        return self._combine(other, operator.add, '{} + {}', False)

    def __radd__(self, other: float) -> CompositionalLoss[Any, Any]:
        """Implements reverse addition for the loss.

        Args:
            other: The float to add to the loss.

        Returns:
            A new CompositionalLoss representing the sum.
        """
        return self.__add__(other)

    def __sub__(
        self,
        other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Subtracts another loss or a float from this loss.

        Args:
            other: The other loss or float to subtract.

        Returns:
            A new CompositionalLoss representing the difference.
        """
        neg_other = other.__neg__()
        return self.__add__(neg_other)

    def __rsub__(
        self, other: float
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Implements reverse subtraction for the loss.

        Args:
            other: The float from which to subtract the loss.

        Returns:
            A new CompositionalLoss representing the difference.
        """
        neg_self = self.__neg__()
        return neg_self.__add__(other)

    def __mul__(
        self,
        other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Multiplies this loss by another loss or a float.

        Args:
            other: The other loss or float to multiply by.

        Returns:
            A new CompositionalLoss representing the product.
        """
        return self._combine(other, operator.mul, '{} x {}')

    def __rmul__(
        self, other: float
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Implements reverse multiplication for the loss.

        Args:
            other: The float to multiply the loss by.

        Returns:
            A new CompositionalLoss representing the product.
        """
        return self.__mul__(other)

    def __truediv__(
        self,
        other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Divides this loss by another loss or a float.

        Args:
            other: The other loss or float to divide by.

        Returns:
            A new CompositionalLoss representing the quotient.
        """
        mul_inv_other = other.__pow__(-1)
        return self.__mul__(mul_inv_other)

    def __rtruediv__(
        self, other: float
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Implements reverse division for the loss.

        Args:
            other: The float to be divided by the loss.

        Returns:
            A new CompositionalLoss representing the quotient.
        """
        mul_inv_self = self.__pow__(-1)
        return mul_inv_self.__mul__(other)

    def __pow__(
        self, other: float
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        """Raises the loss to a given power.

        Args:
            other: The power to raise the loss to.

        Returns:
            A new CompositionalLoss representing the result.
        """

        def _str_other_op(power: float):
            return f'^{power}' if power != 1 else ''

        def _to_floating_point(x: _Tensor) -> _Tensor:
            return x if torch.is_floating_point(x) else x.float()

        if other >= 0:
            higher_is_better = self.higher_is_better
            formula = f'{self.formula}{_str_other_op(other)}'
        else:
            higher_is_better = not self.higher_is_better
            formula = f'1 / {self.formula}{_str_other_op(-other)}'

        return CompositionalLoss(
            criterion=lambda x: _to_floating_point(self.criterion(x)) ** other,
            higher_is_better=higher_is_better,
            formula=formula,
            **self.named_metric_fun,
        )

    def __repr__(self):
        """Returns the string representation of the LossBase object."""
        return f'{self.__class__.__name__}({self.formula})'

    def _check_same_direction(
        self, other: LossBase[_Output_contra, _Target_contra]
    ) -> None:
        """Checks if two losses have the same optimization direction.

        Args:
            other: The other LossBase object to compare with.

        Raises:
            ValueError: If the losses have opposite directions for optimization.
        """
        if self.higher_is_better ^ other.higher_is_better:
            msg = 'Losses {} and {} have opposite directions for optimizations.'
            raise ValueError(msg.format(self, other))

        return

    @staticmethod
    def _remove_outer_parentheses(formula: str) -> str:
        """Removes outer parentheses from a formula string if present.

        Args:
            formula: The formula string.

        Returns:
            The formula string without outer parentheses.
        """
        if formula.startswith('(') and formula.endswith(')'):
            return formula[1:-1]

        if formula.startswith('[]') and formula.endswith(']'):
            return formula[1:-1]

        return formula


class CompositionalLoss(
    LossBase[_Output_contra, _Target_contra],
    MetricCollection[_Output_contra, _Target_contra],
):
    """Loss resulting from an operation between other two losses."""

    def __init__(
        self,
        criterion: Callable[[dict[str, _Tensor]], _Tensor],
        *,
        name='Loss',
        higher_is_better: bool,
        formula: str,
        **named_fun: Callable[[_Output_contra, _Target_contra], _Tensor],
    ) -> None:
        """Constructor.

        Args:
            criterion: function extracting a loss value from metric functions.
            name: identifier for the loss.
            higher_is_better: True if higher values indicate better performance,
                False if lower values are better.
            formula: string representation of the loss formula.
            named_fun: dictionary of named metric functions.
        """
        super().__init__(
            criterion,
            name,
            higher_is_better,
            **named_fun,
            formula=self._format_formula(formula),
        )
        self.higher_is_better = higher_is_better
        return

    @override
    def calculate(
        self: Self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, _Tensor]:
        """Calculates the loss and all associated metric values.

        Args:
            outputs: The model outputs.
            targets: The ground truth targets.

        Returns:
            A dictionary containing the calculated loss and metric values.
        """
        all_metrics = super().calculate(outputs, targets)
        return {self.name: self.criterion(all_metrics)} | all_metrics

    @staticmethod
    def _format_formula(formula: str) -> str:
        """Simplifies the formula string by removing redundant characters.

        Args:
            formula: The formula string.

        Returns:
            The simplified formula string.
        """
        formula = formula.replace('--', '').replace('+ -', '- ')
        if formula.startswith('(') and formula.endswith(')'):
            return formula

        if formula.startswith('[') and formula.endswith(']'):
            return formula

        return '(' + formula + ')'


class Loss(LossBase[_Output_contra, _Target_contra]):
    """Class for a simple loss."""

    def __init__(
        self,
        fun: Callable[[_Output_contra, _Target_contra], _Tensor],
        /,
        *,
        name: str,
        higher_is_better: bool = False,
    ):
        """Constructor.

        Args:
            fun: the callable to calculate the loss.
            name: the name for the loss.
            higher_is_better: the direction for optimization.
        """
        super().__init__(
            operator.itemgetter(name),
            name=name,
            higher_is_better=higher_is_better,
            formula=f'[{name}]',
            **{name: fun},
        )
        self.fun = fun
        return

    @override
    def calculate(
        self, outputs: _Output_contra, targets: _Target_contra
    ) -> dict[str, _Tensor]:
        """Calculates the value of this single loss.

        Args:
            outputs: The model outputs.
            targets: The ground truth targets.

        Returns:
            A dictionary containing the calculated loss value.
        """
        return {self.name: self.fun(outputs, targets)}


def dict_apply(
    dict_fun: dict[str, Callable[[_Output_contra, _Target_contra], _Tensor]],
    outputs: _Output_contra,
    targets: _Target_contra,
) -> dict[str, _Tensor]:
    """Apply the given tensor callables to the provided outputs and targets.

    Args:
        dict_fun: a dictionary of named callables (outputs, targets) -> Tensor.
        outputs: the outputs to apply the tensor callables to.
        targets: the targets to apply the tensor callables to.

    Returns:
        A dictionary containing the resulting values.
    """
    return {
        name: function(outputs, targets) for name, function in dict_fun.items()
    }


def repr_metrics(
    calculator: p.ObjectiveProtocol[Any, Any],
) -> Mapping[str, float]:
    """Represent the metrics as a mapping of named values.

    Args:
        calculator: An ObjectiveProtocol instance from which to compute metrics.

    Returns:
        A mapping of metric names to their float values.
    """
    metrics = calculator.compute()
    if isinstance(metrics, Mapping):
        return {name: value.item() for name, value in metrics.items()}

    if isinstance(metrics, _Tensor):
        return {calculator.__class__.__name__: metrics.item()}

    return {}


class MetricTracker(Generic[_Output_contra, _Target_contra]):
    """Handle metric value tracking and improvement detection.

    This class is responsible for storing metric history, determining
    improvements, and managing patience countdown.

    Note: this class can be used to automatically modify the training strategy.
    Therefore, it does not follow the library conventions for a tracker.

    Attributes:
        metric_name: name of the metric to monitor.
        min_delta: minimum change required to qualify as an improvement.
        patience: number of checks to wait before triggering callback.
        best_is: whether higher or lower values are better.
        filter_fn: function to aggregate recent metric values.
        history: logs of the recorded metrics.
    """

    def __init__(
            self,
            metric_name: str | None = None,
            min_delta: float = 1e-8,
            patience: int = 0,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            filter_fn: Callable[
                [Sequence[float]], float] = operator.itemgetter(-1),
    ) -> None:
        """Constructor.

        Args:
            metric_name: name of the metric to track.
            min_delta: minimum change required to qualify as an improvement.
            patience: number of checks to wait before triggering callback.
            best_is: whether higher or lower metric values are better.
            filter_fn: function to aggregate recent metric values.
        """
        self.metric_name = metric_name
        self.best_is = best_is
        self.filter_fn = filter_fn
        self.min_delta = min_delta
        self._validate_patience(patience)
        self.patience = patience
        self.history = list[float]()
        self._patience_countdown = patience
        self._best_value: float | None = None

    @property
    def best_value(self) -> float:
        """Get the best result observed so far.

        Returns:
            The best filtered value (according to the 'best_is' criterion).

        Raises:
            ResultNotAvailableError: if no results have been logged yet.
        """
        if self._best_value is None:
            try:
                self._best_value = self.history[0]
            except IndexError as ie:
                raise exceptions.ResultNotAvailableError() from ie

        return self._best_value

    @best_value.setter
    def best_value(self, value: float) -> None:
        """Set the best result value."""
        self._best_value = value

    @property
    def filtered_value(self) -> float:
        """Get the current value.

        Returns:
            The current value aggregated from recent ones.

        Raises:
            ResultNotAvailableError: if no results have been logged yet.
        """
        return self.filter_fn(self.history)

    def add_value(self, value: float) -> None:
        """Add a new metric value to the history.

        Args:
            value: The metric value to add.
        """
        self.history.append(value)

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
            self.best_value = aggregated_value
            self._patience_countdown = self.patience
            return True

        self._patience_countdown -= 1
        return False

    def is_patient(self) -> bool:
        """Check whether to be patient."""
        return self._patience_countdown > 0

    def reset_patience(self) -> None:
        """Reset patience countdown to the maximum."""
        self._patience_countdown = self.patience

    @staticmethod
    def _validate_patience(patience: int) -> None:
        if patience < 0:
            raise ValueError('Patience must be a non-negative integer.')

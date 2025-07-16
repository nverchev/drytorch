"""
Module containing classes to create and combine loss and metrics.

The interface is similar to https://github.com/Lightning-AI/torchmetrics,
with stricter typing and simpler construction. MetricCollection and
CompositionalMetric from torchmetrics change their state; here a functional
approach is preferred.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Hashable
import copy
import operator
from typing import Mapping, Optional, Self, TypeVar
from typing_extensions import override
import warnings

import torch

from drytorch import exceptions
from drytorch import protocols as p
from drytorch.utils import statistics

_K = TypeVar('_K', bound=Hashable)
_V = TypeVar('_V')

_Output_contra = TypeVar('_Output_contra',
                         bound=p.OutputType,
                         contravariant=True)
_Target_contra = TypeVar('_Target_contra',
                         bound=p.TargetType,
                         contravariant=True)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)

_Tensor = torch.Tensor


class Objective(
    p.ObjectiveProtocol[_Output_contra, _Target_contra],
    metaclass=abc.ABCMeta
):
    def __init__(
            self,
            **metric_fun: Callable[
                [_Output_contra, _Target_contra], _Tensor,
            ],
    ) -> None:
        self._aggregator = statistics.TorchAverager()
        self.named_metric_fun = metric_fun

    @override
    def compute(self: Self) -> dict[str, _Tensor]:
        if not self._aggregator:
            warnings.warn(exceptions.ComputedBeforeUpdatedWarning(self))

        return self._aggregator.reduce()

    @override
    def update(self: Self,
               outputs: _Output_contra,
               targets: _Target_contra) -> dict[str, _Tensor]:
        results = self.calculate(outputs, targets)
        self._aggregator += {key: value.detach()
                             for key, value in results.items()}
        return results

    @override
    def reset(self: Self) -> None:
        self._aggregator.clear()
        return

    @abc.abstractmethod
    def calculate(self: Self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, _Tensor]:
        """
        Method responsible for the calculations.

        Args:
            outputs: model outputs.
            targets: ground truth.
        """

    def copy(self) -> Self:
        """Create a (deep)copy of self."""
        return self.__deepcopy__({})

    def merge_state(self: Self, other: Self) -> None:
        """
        Merge metric states.

        Args:
            other: metric to be merged with.
        """
        self._aggregator += other._aggregator
        return

    def __or__(
            self,
            other: Objective[_Output_contra, _Target_contra]
    ) -> MetricCollection[_Output_contra, _Target_contra]:
        named_metric_fun = self.named_metric_fun | other.named_metric_fun
        return MetricCollection(**named_metric_fun)

    def __deepcopy__(self, memo: dict) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            result.__dict__[k] = copy.deepcopy(v, memo)

        return result


class MetricCollection(Objective[_Output_contra, _Target_contra]):

    @override
    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, _Tensor]:
        return dict_apply(self.named_metric_fun, outputs, targets)


class Metric(Objective[_Output_contra, _Target_contra]):

    def __init__(self,
                 fun: Callable[[_Output_contra, _Target_contra], _Tensor],
                 /,
                 *,
                 name: str,
                 higher_is_better: Optional[bool] = None) -> None:
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
    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, _Tensor]:
        return {self.name: self.fun(outputs, targets)}


class LossBase(
    MetricCollection[_Output_contra, _Target_contra],
    p.LossCalculatorProtocol[_Output_contra, _Target_contra],
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
    def forward(self,
                outputs: _Output_contra,
                targets: _Target_contra) -> _Tensor:
        metrics = self.update(outputs, targets)
        return self.criterion(metrics).mean()

    def __or__(
            self,
            other: Objective[_Output_contra, _Target_contra]
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        named_metric_fun = self.named_metric_fun | other.named_metric_fun
        return CompositionalLoss(criterion=self.criterion,
                                 name=self.name,
                                 higher_is_better=self.higher_is_better,
                                 formula=self.formula,
                                 **named_metric_fun)

    def _combine(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
            operation: Callable[[_Tensor, _Tensor], _Tensor],
            op_fmt: str,
            requires_parentheses: bool = True,
    ) -> CompositionalLoss[_Output_contra, _Target_contra]:
        if isinstance(other, LossBase):
            named_metric_fun = self.named_metric_fun | other.named_metric_fun
            str_first = self.formula
            str_second = other.formula

            # apply should combine losses that share the same direction
            self._check_same_direction(other)

            def _combined(x: dict[str, _Tensor]) -> _Tensor:
                return operation(self.criterion(x), other.criterion(x))

        elif isinstance(other, (float, int)):
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
        return CompositionalLoss(criterion=_combined,
                                 higher_is_better=self.higher_is_better,
                                 formula=formula,
                                 **named_metric_fun)

    def __neg__(self) -> CompositionalLoss:
        return CompositionalLoss(criterion=lambda x: -self.criterion(x),
                                 higher_is_better=not self.higher_is_better,
                                 formula=f'-{self.formula}',
                                 **self.named_metric_fun)

    def __add__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        return self._combine(other, operator.add, '{} + {}', False)

    def __radd__(self, other: float) -> CompositionalLoss:
        return self.__add__(other)

    def __sub__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        neg_other = other.__neg__()
        return self.__add__(neg_other)

    def __rsub__(self, other: float) -> CompositionalLoss:
        neg_self = self.__neg__()
        return neg_self.__add__(other)

    def __mul__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        return self._combine(other, operator.mul, '{} x {}')

    def __rmul__(self, other: float) -> CompositionalLoss:
        return self.__mul__(other)

    def __truediv__(
            self,
            other: LossBase[_Output_contra, _Target_contra] | float,
    ) -> CompositionalLoss:
        mul_inv_other = other.__pow__(-1)
        return self.__mul__(mul_inv_other)

    def __rtruediv__(self, other: float) -> CompositionalLoss:
        mul_inv_self = self.__pow__(-1)
        return mul_inv_self.__mul__(other)

    def __pow__(self, other: float) -> CompositionalLoss:

        def _str_other_op(power: float):
            return f'^{power}' if power != 1 else ''

        if other >= 0:
            higher_is_better = self.higher_is_better
            formula = f'{self.formula}{_str_other_op(other)}'
        else:
            higher_is_better = not self.higher_is_better
            formula = f'1 / {self.formula}{_str_other_op(-other)}'

        return CompositionalLoss(criterion=lambda x: self.criterion(x) ** other,
                                 higher_is_better=higher_is_better,
                                 formula=formula,
                                 **self.named_metric_fun)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.formula})'

    def _check_same_direction(self, other: LossBase) -> None:
        if self.higher_is_better ^ other.higher_is_better:
            msg = 'Losses {} and {} have opposite directions for optimizations.'
            raise ValueError(msg.format(self, other))

        return

    @staticmethod
    def _remove_outer_parentheses(formula: str) -> str:
        if formula.startswith('(') and formula.endswith(')'):
            return formula[1:-1]

        return formula


class CompositionalLoss(
    LossBase[_Output_contra, _Target_contra],
    MetricCollection[_Output_contra, _Target_contra]
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
            formula=f'({self._simplify_formula(formula)})',
        )
        self.higher_is_better = higher_is_better
        return

    def __repr__(self):
        return f'{self.__class__.__name__}{self.formula}'

    @override
    def calculate(self: Self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, _Tensor]:
        all_metrics = super().calculate(outputs, targets)
        return {self.name: self.criterion(all_metrics)} | all_metrics

    @staticmethod
    def _simplify_formula(formula: str) -> str:
        formula = formula.replace('--', '').replace('+ -', '- ')
        if formula.startswith('(') and formula.endswith(')'):
            formula = formula[1:-1]
        return formula


class Loss(LossBase[_Output_contra, _Target_contra]):
    """Class for a simple loss."""

    def __init__(
            self,
            fun: Callable[[_Output_contra, _Target_contra], _Tensor],
            /,
            *,
            name: str,
            higher_is_better: bool = False):
        """Constructor.
        
        Args:
            fun: the callable to calculate the loss.
            name: the name for the loss.
            higher_is_better: the direction for optimization.
        """
        super().__init__(operator.itemgetter(name),
                         name=name,
                         higher_is_better=higher_is_better,
                         formula=f'[{name}]',
                         **{name: fun})
        self.fun = fun
        return

    @override
    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> dict[str, _Tensor]:
        return {self.name: self.fun(outputs, targets)}


def dict_apply(
        dict_fun: dict[str, Callable[
            [_Output_contra, _Target_contra], _Tensor]
        ],
        outputs: _Output_contra,
        targets: _Target_contra,
) -> dict[str, _Tensor]:
    """
    Apply the given tensor callables to the provided outputs and targets.

    Args:
        dict_fun: a dictionary of named callables (outputs, targets) -> Tensor.
        outputs: the outputs to apply the tensor callables to.
        targets: the targets to apply the tensor callables to.

    Returns:
        A dictionary containing the resulting values.
    """
    return {name: function(outputs, targets)
            for name, function in dict_fun.items()}


def repr_metrics(calculator: p.ObjectiveProtocol) -> Mapping[str, float]:
    """Represent the metrics as a mapping of named values."""
    metrics = calculator.compute()
    if isinstance(metrics, Mapping):
        return {name: value.item() for name, value in metrics.items()}

    if isinstance(metrics, _Tensor):
        return {calculator.__class__.__name__: metrics.item()}

    return {}

from typing import TypeVar, Hashable, Optional, Self

import torch
from dry_torch import protocols as p
from dry_torch import exceptions

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


class MetricsCalculator(
    p.MetricsCalculatorProtocol[_Output_contra, _Target_contra]
):

    def __init__(
            self,
            **metric_fun: p.TensorCallable[_Output_contra, _Target_contra],
    ) -> None:
        self.named_metric_fun = metric_fun
        self._metrics: Optional[dict[str, torch.Tensor]] = None

    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> None:
        self._metrics = {name: function(outputs, targets)
                         for name, function in self.named_metric_fun.items()}

    @property
    def metrics(self: Self) -> dict[str, torch.Tensor]:
        if self._metrics is None:
            raise exceptions.AccessBeforeCalculateError()
        return self._metrics

    def reset_calculated(self) -> None:
        self._metrics = None


class LossCalculator(
    MetricsCalculator[_Output_contra, _Target_contra],
    p.LossCalculatorProtocol[_Output_contra, _Target_contra],
):

    def __init__(
            self,
            loss_fun: p.TensorCallable[_Output_contra, _Target_contra],
            **metric_fun: p.TensorCallable[_Output_contra, _Target_contra],
    ) -> None:
        super().__init__(**metric_fun)
        self.loss_fun = loss_fun
        self._criterion: Optional[torch.Tensor] = None

    def calculate(self,
                  outputs: _Output_contra,
                  targets: _Target_contra) -> None:
        super().calculate(outputs, targets)
        criterion = self.loss_fun(outputs, targets)
        self._metrics: dict[str, torch.Tensor]
        self._metrics.update(criterion=criterion)
        self._criterion = criterion.mean()

        return

    @property
    def criterion(self) -> torch.Tensor:
        if self._criterion is None:
            raise exceptions.AccessBeforeCalculateError()
        return self._criterion

    def reset_calculated(self) -> None:
        super().reset_calculated()
        self._criterion = None

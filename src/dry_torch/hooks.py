from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar, Optional, ParamSpec, Literal

import numpy as np
import numpy.typing as npt

from src.dry_torch import exceptions
from src.dry_torch import protocols as p
from src.dry_torch import events

_Class = TypeVar('_Class')
_P = ParamSpec('_P')


class HookRegistry(Generic[_Class]):

    def __init__(self) -> None:
        self._hooks: list[Callable[[_Class], None]] = []

    def register(self, hook: Callable[[_Class], None]) -> None:
        self._hooks.append(hook)
        return

    def register_all(self, hook_list: list[Callable[[_Class], None]]) -> None:
        for hook in hook_list:
            self.register(hook)
        return

    def execute(self, class_instance: _Class) -> None:
        for hook in self._hooks:
            hook(class_instance)
        return


def saving_hook() -> Callable[[p.TrainerProtocol], None]:
    def call(instance: p.TrainerProtocol) -> None:
        instance.save_checkpoint()

    return call


def static_hook(
        hook: Callable[[], None]
) -> Callable[[p.TrainerProtocol], None]:
    def call(_: p.TrainerProtocol) -> None:
        hook()

    return call


def static_hook_closure(
        static_closure: Callable[_P, Callable[[], None]]
) -> Callable[_P, Callable[[p.TrainerProtocol], None]]:
    def closure_hook(
            *args: _P.args,
            **kwargs: _P.kwargs,
    ) -> Callable[[p.TrainerProtocol], None]:
        static_callable = static_closure(*args, **kwargs)

        def call(_: p.TrainerProtocol) -> None:
            nonlocal static_callable
            return static_callable()

        return call

    return closure_hook


def call_every(
        interval: int,
        hook: Callable[[p.TrainerProtocol], None],
        start: int = 0,
) -> Callable[[p.TrainerProtocol], None]:
    def call(instance: p.TrainerProtocol) -> None:
        epoch = instance.model.epoch
        if epoch % interval == start or instance.terminated:
            hook(instance)

    return call


class EarlyStoppingCallback:
    def __init__(
            self,
            metric_name: Optional[str] = None,
            monitor_validation: bool = True,
            min_delta: float = 0.,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[npt.NDArray], float]] = None,
            pruning: Optional[dict[int, float]] = None,
            start_from_epoch: int = 0,
            monitor_external: Optional[p.EvaluationProtocol] = None,
    ) -> None:
        self.metric_name = metric_name
        self.monitor_validation = monitor_validation
        self.min_delta = min_delta
        self.patience = patience
        self.best_is = best_is
        self.pruning = pruning
        self.start_from_epoch = start_from_epoch
        self.monitor_external = monitor_external
        self.monitor_log: deque[float] = deque(maxlen=patience + 1)
        self.pruning_results: dict[int, float] = {}

        default_aggregate: Callable[[npt.NDArray], float]
        if best_is == 'lower':
            self.best_result = float('inf')
            default_aggregate = np.min
        elif best_is == 'higher':
            self.best_result = float('-inf')
            default_aggregate = np.max
        else:
            self.best_result = 0
            default_aggregate = np.mean
        if aggregate_fn is None:
            self.aggregate_fn = default_aggregate
        else:
            self.aggregate_fn = aggregate_fn

    def __call__(self, instance: p.TrainerProtocol) -> None:
        if self.monitor_validation:
            if instance.validation is None:
                raise exceptions.NoValidationError
            monitor: p.EvaluationProtocol = instance.validation
        else:
            monitor = instance if self.monitor_external is None else (
                self.monitor_external)
        current_epoch = instance.model.epoch
        last_metrics = monitor.metrics
        if self.metric_name is None:
            self.metric_name = list(last_metrics.keys())[0]
        elif self.metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(monitor.name, self.metric_name)
        self.monitor_log.append(last_metrics[self.metric_name])
        if current_epoch < max(self.start_from_epoch, self.patience):
            return
        if self.best_is == 'auto':
            if len(self.monitor_log) > 1:
                if self.monitor_log[0] > self.monitor_log[-1]:
                    self.best_is = 'lower'
                    self.best_result = float('inf')
                    self.aggregate_fn = np.min
                else:
                    self.best_is = 'higher'
                    self.best_result = float('-inf')
                    self.aggregate_fn = np.max
            return

        aggregate_result = self.aggregate_fn(np.array(self.monitor_log))
        if self.best_is == 'lower':
            condition = self.best_result + self.min_delta < aggregate_result
            if self.pruning is not None and current_epoch in self.pruning:
                condition |= self.pruning[current_epoch] <= aggregate_result
                self.pruning_results[current_epoch] = aggregate_result
        else:
            condition = self.best_result - self.min_delta > aggregate_result
            if self.pruning is not None and current_epoch in self.pruning:
                condition |= self.pruning[current_epoch] >= aggregate_result
                self.pruning_results[current_epoch] = aggregate_result

        if condition:
            instance.terminate_training()
            events.TerminatedTraining(instance.model.epoch, 'early stopping')

        else:
            self.best_result = aggregate_result
        return

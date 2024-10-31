from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar, Optional, ParamSpec

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
    def closure_hook(*args: _P.args,
                     **kwargs: _P.kwargs) -> Callable[[p.TrainerProtocol],
    None]:
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


def early_stopping_callback(
        metric_name: Optional[str] = None,
        monitor_validation: bool = True,
        aggregate_fun: Callable[[npt.NDArray], float] = np.min,
        min_delta: float = 0.,
        patience: int = 10,
        lower_is_best: bool = True,
        baseline: Optional[float] = None,
        start_from_epoch: int = 0,
        monitor_external: Optional[p.EvaluationProtocol] = None
) -> Callable[[p.TrainerProtocol], None]:
    best_result = float('inf') if lower_is_best else 0
    monitor_log: deque[float] = deque(maxlen=patience + 1)

    def call(instance: p.TrainerProtocol):
        nonlocal metric_name, best_result, monitor_log
        if monitor_validation:
            if instance.validation is None:
                raise exceptions.NoValidationError
            monitor: p.EvaluationProtocol = instance.validation
        else:
            monitor = instance if monitor_external is None else monitor_external
        last_metrics = monitor.metrics
        if metric_name is None:
            metric_name = list(last_metrics.keys())[0]
        elif metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(monitor.name, metric_name)
        monitor_log.append(last_metrics[metric_name])
        if instance.model.epoch < max(start_from_epoch, patience):
            return
        aggregated_result = aggregate_fun(np.array(monitor_log))
        if lower_is_best:
            if baseline is None:
                condition = best_result + min_delta < aggregated_result
            else:
                condition = baseline + min_delta <= aggregated_result
        else:
            if baseline is None:
                condition = best_result - min_delta > aggregated_result
            else:
                condition = baseline - min_delta >= aggregated_result

        if condition:
            instance.terminate_training()
            events.TerminatedTraining(instance.model.epoch, 'early stopping')

        else:
            best_result = aggregated_result
        return

    return call

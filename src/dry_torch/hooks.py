import logging
from typing import Generic, TypeVar, Callable, Optional

import numpy as np
import numpy.typing as npt

from dry_torch import descriptors
from dry_torch import exceptions
from dry_torch import io
from dry_torch import log_settings
from dry_torch import protocols as p
from dry_torch import tracking
from typing_extensions import ParamSpec

_Class = TypeVar('_Class')
_P = ParamSpec('_P')

logger = logging.getLogger('dry_torch')



class HookRegistry(Generic[_Class]):

    def __init__(self) -> None:
        self._hooks: list[Callable[[_Class], None]] = []

    def register(self, hook: Callable[[_Class], None]) -> None:
        self._hooks.append(hook)
        return

    def execute(self, class_instance: _Class) -> None:
        for hook in self._hooks:
            hook(class_instance)
        return


def saving_hook(
        replace_previous: bool = False
) -> Callable[[p.TrainerProtocol], None]:
    def call(instance: p.TrainerProtocol) -> None:
        instance.save_checkpoint(replace_previous=replace_previous)

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
        epoch = tracking.track(instance.model).epoch
        if epoch % interval == start or instance.terminated:
            hook(instance)

    return call


def early_stopping_callback(
        metric_name: str,
        monitor_dataset: descriptors.Split = descriptors.Split.VAL,
        aggregate_fun: Callable[[npt.NDArray], float] = np.min,
        min_delta: float = 0.,
        patience: int = 10,
        lower_is_best: bool = True,
        baseline: Optional[float] = None,
        start_from_epoch: int = 0,
) -> Callable[[p.TrainerProtocol], None]:
    best_result = float('inf') if lower_is_best else 0

    def call(instance: p.TrainerProtocol):
        nonlocal best_result
        model_tracker = tracking.track(instance.model)
        if model_tracker.epoch < max(start_from_epoch, patience):
            return
        monitor_log = model_tracker.log[monitor_dataset]
        if metric_name not in monitor_log:
            raise exceptions.MetricNotFoundError(metric_name,
                                                 monitor_dataset.name)
        metric_results = monitor_log[metric_name]
        last_results = metric_results[
            monitor_log['Epoch'] >= model_tracker.epoch - patience
            ]
        aggregated_result = aggregate_fun(last_results)
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
            logger.log(log_settings.INFO_LEVELS.metrics,
                       'Terminated by early stopping.')
        else:
            best_result = aggregated_result
        return

    return call

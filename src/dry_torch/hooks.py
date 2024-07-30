import logging
from typing import Generic, TypeVar, Callable, Optional

from dry_torch import descriptors
from dry_torch import protocols as p
from dry_torch import tracking
from dry_torch import log_settings

_Class = TypeVar('_Class')

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


def validation_hook(instance: p.TrainerProtocol) -> None:
    instance.validate()
    return


def early_stopping_callback(
        monitor_dataset: descriptors.Split = descriptors.Split.VAL,
        metric: str = 'Criterion',
        min_delta: float = 0.,
        patience: int = 10,
        lower_is_best: bool = True,
        baseline: Optional[float] = None,
        start_from_epoch: int = 0,
) -> Callable[[p.TrainerProtocol], None]:
    best_result = float('inf') if lower_is_best else 0

    def _early_stopping(instance: p.TrainerProtocol):
        nonlocal best_result
        exp = tracking.Experiment.current()
        model_tracker = exp.tracker[instance.model.name]
        if model_tracker.epoch < start_from_epoch:
            return

        log = model_tracker.log[monitor_dataset]
        last_results = log[metric][-(patience + 1):]
        best_last_result = min(last_results)
        if baseline is None:
            condition = best_result + min_delta <= best_last_result
        else:
            condition = baseline + min_delta <= best_last_result

        if not lower_is_best:
            condition = not condition

        if condition:
            instance.terminate_training()
            logger.log(log_settings.INFO_LEVELS.metrics,
                       'Terminated by early stopping.')
        else:
            best_result = best_last_result
        return

    return _early_stopping

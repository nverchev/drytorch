from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar, Optional, ParamSpec, Literal

import numpy as np
import numpy.typing as npt
from zmq.backend import first

from src.dry_torch import exceptions
from src.dry_torch import protocols as p
from src.dry_torch import log_events

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
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 0.,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[npt.NDArray], float]] = None,
            pruning: Optional[dict[int, float]] = None,
            start_from_epoch: int = 0,
    ) -> None:
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.patience = patience
        self.best_is = best_is
        self.pruning = pruning
        self.start_from_epoch = start_from_epoch
        self.monitor = monitor
        self.monitor_log: deque[float] = deque(maxlen=patience + 1)
        self._best_result: Optional[float] = None
        self._aggregate_fn = aggregate_fn


    @property
    def best_result(self) ->  float:
        if self._best_result is None:
            first_result = self.monitor_log[0]
            self._best_result = first_result
            return first_result
        return self._best_result

    @best_result.setter
    def best_result(self, value: float) -> None:
        self._best_result = value
        return

    @property
    def aggregate_fn(self) ->  Callable[[npt.NDArray], float]:
        if self._aggregate_fn is None:
            if self.best_is == 'lower':
                self._aggregate_fn = np.min
                return np.min
            elif self._best_result == 'higher':
                self._aggregate_fn = np.max
                return np.max
            else:
                return lambda array: array[-1]
        return self._aggregate_fn

    def __call__(self, instance: p.TrainerProtocol) -> None:
        current_epoch = instance.model.epoch
        if current_epoch < max(self.start_from_epoch, self.patience):
            return

        if self.monitor is None:
            if instance.validation is None:
                monitor: p.EvaluationProtocol = instance
            else:
                monitor = instance.validation
        else:
            monitor = self.monitor

        last_metrics = monitor.metrics
        if self.metric_name is None:
            self.metric_name = list(last_metrics.keys())[0]
        elif self.metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(monitor.name, self.metric_name)
        self.monitor_log.append(last_metrics[self.metric_name])

        if len(self.monitor_log) == 1:
            return

        aggregate_result = self.aggregate_fn(np.array(self.monitor_log))

        if self.best_is == 'auto':
            if self.best_result > aggregate_result:
                self.best_is = 'lower'  # at start best_result is worse result
            else:
                self.best_is = 'higher'
            condition = False
        elif self.best_is == 'lower':
            condition = self.best_result + self.min_delta < aggregate_result
            if self.pruning is not None and current_epoch in self.pruning:
                condition |= self.pruning[current_epoch] <= aggregate_result
        else:
            condition = self.best_result - self.min_delta > aggregate_result
            if self.pruning is not None and current_epoch in self.pruning:
                condition |= self.pruning[current_epoch] >= aggregate_result

        if condition:
            instance.terminate_training()
            log_events.TerminatedTraining(instance.model.epoch,
                                          'early stopping')
        else:
            self.best_result = aggregate_result
        return

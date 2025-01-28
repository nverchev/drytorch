"""Registry and hooks for a class following the Trainer protocol."""

from collections import deque
from collections.abc import Callable, Sequence
import functools
from typing import Generic, Literal, Optional, ParamSpec, TypeAlias, TypeVar
from typing import cast

from src.dry_torch import calculating
from src.dry_torch import log_events
from src.dry_torch import exceptions
from src.dry_torch import protocols as p

_T = TypeVar('_T')
_P = ParamSpec('_P')
_Hook: TypeAlias = Callable[[p.TrainerProtocol], None]


class HookRegistry(Generic[_T]):
    """
    A registry for managing and executing hooks.

    The hooks have a generic object as input and can access it.

    Attributes:
        _hooks: A list of registered hooks.
    """

    def __init__(self) -> None:
        """
        Initializes the HookRegistry with an empty list of hooks.
        """
        self._hooks: list[Callable[[_T], None]] = []

    def register(self, hook: Callable[[_T], None]) -> None:
        """
        Registers a single hook.

        Args:
            hook: The hook to register.
        """
        self._hooks.append(hook)
        return

    def register_all(self, hook_list: list[Callable[[_T], None]]) -> None:
        """
        Registers multiple hooks.

        Args:
            hook_list: The list of hooks to register.
        """
        for hook in hook_list:
            self.register(hook)
        return

    def execute(self, input_object: _T) -> None:
        """
        Executes the registered hooks in order of registration.

        Args:
            input_object: The input to pass to each hook.
        """
        for hook in self._hooks:
            hook(input_object)
        return


def saving_hook() -> _Hook:
    """
    Creates a hook that saves the model's checkpoint.

    Returns:
        A callable hook that saves a checkpoint.
    """

    def _call(instance: p.TrainerProtocol) -> None:
        instance.save_checkpoint()

    return _call


def static_hook(hook: Callable[[], None]) -> _Hook:
    """
    Wraps a static callable as a hook.

    Args:
        hook: The static callable to wrap.

    Returns:
        A hook that invokes the static callable.
    """

    @functools.wraps(hook)
    def _call(_: p.TrainerProtocol) -> None:
        hook()

    return _call


def static_hook_closure(static_closure: Callable[_P, Callable[[], None]]
                        ) -> Callable[_P, _Hook]:
    """
    Creates a hook from a static closure.

    Args:
        static_closure: The static closure to wrap.

    Returns:
        A hook that invokes the static callable.
    """

    @functools.wraps(static_closure)
    def _closure_hook(*args: _P.args, **kwargs: _P.kwargs) -> _Hook:
        static_callable = static_closure(*args, **kwargs)

        def _call(_: p.TrainerProtocol) -> None:
            nonlocal static_callable
            return static_callable()

        return _call

    return _closure_hook


def call_every(interval: int, hook: _Hook, start: int = 0) -> _Hook:
    """
    Creates a hook that executes at specified intervals.

    Args:
        interval: The interval (in epochs) at which the hook is called.
        hook: The hook to execute.
        start: The starting epoch offset. Defaults to 0.

    Returns:
        A callable hook.
    """

    @functools.wraps(hook)
    def _call(instance: p.TrainerProtocol) -> None:
        epoch = instance.model.epoch
        if epoch % interval == start or instance.terminated:
            hook(instance)

    return _call


class EarlyStoppingCallback:
    """
    Implements early stopping logic for training models.
    """

    def __init__(
            self,
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 0.,
            patience: int = 10,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[Sequence], float]] = None,
            pruning: Optional[dict[int, float]] = None,
            start_from_epoch: int = 2,
    ) -> None:
        """
        Initializes the EarlyStoppingCallback.

        Args:
        metric_name: The name of the metric to monitor. Defaults to first found.
        monitor: The evaluation protocol to monitor. Defaults to validation.
            Falls back to the trainer class.
        min_delta: The minimum change in metric to qualify as an improvement.
        patience: The number of calls to wait before stopping. Defaults to 10.
        best_is: Determines if higher or lower values are better. Defaults to
            automatic inference.
        aggregate_fn: Function to aggregate recent metric values. Defaults to
            min or max depending on best_is.
        pruning: A mapping of epoch numbers to pruning thresholds.
        start_from_epoch: The earliest epoch to stop. Defaults to 2.
        """
        if metric is None or isinstance(metric, str):
            self._metric_name = metric
        elif name := getattr(metric, 'name', False):
            self._metric_name = str(name)
        else:
            self._metric_name = metric.__class__.__name__
        self._min_delta = min_delta
        self._patience = patience
        higher_is_better = getattr(metric, 'higher_is_better', None)
        if higher_is_better is True:
            self._best_is = 'higher'
        elif higher_is_better is False:
            self._best_is = 'lower'
        else:
            self._best_is = best_is
        self._pruning = pruning
        self._start_from_epoch = start_from_epoch
        self._monitor = monitor
        self._monitor_log = deque[float](maxlen=patience + 1)
        self._best_result: Optional[float] = None
        self._aggregate_fn = aggregate_fn

    @property
    def best_result(self) -> float:
        """
        Returns the best observed result.

        Returns:
            The best observed metric value.
        """
        if self._best_result is None:
            try:
                first_result = self._monitor_log[0]
            except IndexError:
                raise exceptions.ResultNotAvailableError()
            self._best_result = first_result
            return first_result
        return self._best_result

    @best_result.setter
    def best_result(self, value: float) -> None:
        """
        Sets the best result.

        Args:
            The new best result value.
        """
        self._best_result = value
        return

    @property
    def aggregate_fn(self) -> Callable[[Sequence], float]:
        """
        Returns the aggregation function used for metrics.

        Returns:
            The aggregation function.
        """
        if self._aggregate_fn is None:
            if self._best_is == 'lower':
                self._aggregate_fn = min
                return min
            elif self._best_is == 'higher':
                self._aggregate_fn = max
                return max
            else:
                return lambda array: array[0]
        return self._aggregate_fn

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Evaluates whether training should be stopped early.

        Args:
            instance: The Trainer instance to evaluate.

        Raises:
            MetricNotFoundError: If the specified metric is not found.
        """
        log_events.TerminatedTraining(instance.model.epoch, 'early stopping')
        if self._monitor is None:
            if instance.validation is None:
                monitor = cast(p.EvaluationProtocol, instance)
            else:
                monitor = instance.validation
        else:
            monitor = self._monitor

        last_metrics = calculating.repr_metrics(monitor.calculator)

        if self._metric_name is None:
            self._metric_name = list(last_metrics.keys())[0]
        elif self._metric_name not in last_metrics:
            raise exceptions.MetricNotFoundError(monitor.name,
                                                 self._metric_name)
        if last_metrics != last_metrics:
            instance.terminate_training()
            log_events.TerminatedTraining(instance.model.epoch,
                                          f'Metric is not a number.')

        self._monitor_log.append(last_metrics[self._metric_name])

        current_epoch = instance.model.epoch

        if (current_epoch < self._start_from_epoch or
                len(self._monitor_log) <= self._patience):
            return

        aggregate_result = self.aggregate_fn(self._monitor_log)

        if self._best_is == 'auto':
            if self._monitor_log[0] > aggregate_result:
                self._best_is = 'lower'  # start result is worse result
            else:
                self._best_is = 'higher'
            condition = False
        elif self._best_is == 'lower':
            condition = self.best_result + self._min_delta <= aggregate_result
            if self._pruning is not None and current_epoch in self._pruning:
                condition |= self._pruning[current_epoch] <= aggregate_result
        else:
            condition = self.best_result - self._min_delta >= aggregate_result
            if self._pruning is not None and current_epoch in self._pruning:
                condition |= self._pruning[current_epoch] >= aggregate_result

        if condition:
            instance.terminate_training()
            log_events.TerminatedTraining(instance.model.epoch,
                                          'early stopping')
        else:
            self.best_result = aggregate_result
        return

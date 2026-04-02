"""Module containing schedulers for the learning rates."""

from __future__ import annotations

import abc
import dataclasses

from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import numpy as np

from typing_extensions import override

from drytorch.core import protocols as p


__all__ = [
    'AbstractScheduler',
    'ConstantScheduler',
    'CosineScheduler',
    'ExponentialScheduler',
    'PolynomialScheduler',
    'RescaleScheduler',
    'RestartScheduler',
    'StepScheduler',
    'WarmupScheduler',
    'rescale',
    'restart',
    'warmup',
]

_SchedulingLogic: TypeAlias = Callable[[float, int], float]


class AbstractScheduler(p.SchedulerProtocol, abc.ABC):
    """Abstract class for the scheduler.

    Attributes:
        base_scheduler_name: name of the base scheduler for representation.
        parameters: metadata associated with the scheduler.
    """

    base_scheduler_name: str
    parameters: dict[str, Any]

    def __call__(self, base_lr: float, epoch: int) -> float:
        """Modify the learning rate according to a schedule.

        Args:
            base_lr: initial learning rate.
            epoch: the current epoch.

        Returns:
            scheduled value for the learning rate.

        Raises:
            ValueError: if base_lr or epoch are non-positive.
        """
        if base_lr < 0 or epoch < 0:
            raise ValueError('Base learning rate and epoch must be positive.')

        return self._compute(base_lr, epoch)

    def bind(
        self,
        f: Callable[[_SchedulingLogic], AbstractScheduler],
        /,
    ) -> ComposedScheduler:
        """Allow transformation of the scheduler.

        Args:
            f: a function specifying the transformation.

        Returns:
            the transformed scheduler.
        """
        next_scheduler = f(self._compute)
        parameters = self.parameters | next_scheduler.parameters
        return ComposedScheduler(
            self.base_scheduler_name, parameters, next_scheduler._compute
        )

    @abc.abstractmethod
    def _compute(self, base_lr: float, epoch: int) -> float:
        """Compute the scheduled value.

        Args:
            base_lr: value when epoch is 0.
            epoch: variable of the function.

        Returns:
            the value for learning rate to use.
        """


class ComposedScheduler(AbstractScheduler):
    """A scheduler produced by composing transformations.

    Attributes:
        base_scheduler_name: name of the base scheduler for representation.
        parameters: merged parameters from all composed schedulers.
    """

    def __init__(
        self,
        base_scheduler: str,
        parameters: dict[str, Any],
        logic: _SchedulingLogic,
    ):
        """Initialize.

        Args:
            base_scheduler: name of the base scheduler for representation.
            logic: the composed scheduling callable.
            parameters: merged parameters from all composed schedulers.
        """
        self.base_scheduler_name = base_scheduler
        self._logic = logic
        self.parameters = parameters
        return

    @override
    def _compute(self, base_lr: float, epoch: int) -> float:
        return self._logic(base_lr, epoch)


class TransformScheduler(AbstractScheduler, abc.ABC):
    """Base class for scheduler transformations.

    Attributes:
        logic: callable to calculate the scheduling.
        parameters: metadata associated with the scheduler.
        base_scheduler_name: name of the base scheduler for representation.
    """

    def __init__(self, logic: _SchedulingLogic):
        """Initialize.

        Args:
            logic: callable to calculate the scheduling.
        """
        self.logic = logic
        self.parameters = {}
        self.base_scheduler_name = self.__class__.__name__
        return

    @abc.abstractmethod
    def _compute(self, base_lr: float, epoch: int) -> float:
        """Compute the scheduled value.

        Args:
            base_lr: value when epoch is 0.
            epoch: variable of the function.

        Returns:
            the value for learning rate to use.
        """


class RescaleScheduler(TransformScheduler):
    """Scheduler adding scaling to existing logic.

    Attributes:
        logic: callable to calculate the scheduling.
        factor: factor to multiply the output by.
        parameters: metadata associated with the scheduler.
        base_scheduler_name: name of the base scheduler for representation.
    """

    def __init__(self, logic: _SchedulingLogic, factor: float):
        """Initialize.

        Args:
            logic: callable to calculate the scheduling.
            factor: factor to multiply the output by.
        """
        super().__init__(logic)
        if factor <= 0:
            raise ValueError('factor must be positive.')

        self.factor = factor
        self.parameters['factor'] = factor
        return

    def _compute(self, start_val: float, epoch: int) -> float:
        return self.factor * self.logic(start_val, epoch)


class RestartScheduler(TransformScheduler):
    """Scheduler adding periodic restarts to existing logic.

    Attributes:
        logic: callable to calculate the scheduling.
        restart_interval: number of epochs between restarts.
        restart_fraction: fraction to use when restarting.
        max_restart: maximum number of restarts before deactivating.
        parameters: metadata associated with the scheduler.
        base_scheduler_name: name of the base scheduler for representation.
    """

    def __init__(
        self,
        logic: _SchedulingLogic,
        restart_interval: int,
        restart_fraction: float = 1.0,
        max_restart: int | None = None,
    ):
        """Initialize.

        Args:
            logic: callable to calculate the scheduling.
            restart_interval: number of epochs between restarts.
            restart_fraction: fraction to use when restarting.
            max_restart: maximum number of restarts before deactivating.
        """
        super().__init__(logic)
        if restart_interval <= 0:
            raise ValueError('restart_interval must be positive.')

        if restart_fraction <= 0:
            raise ValueError('restart_fraction must be positive.')

        if max_restart is not None and max_restart <= 0:
            raise ValueError('max_restart must be positive.')

        self.restart_interval = restart_interval
        self.restart_fraction = restart_fraction
        self.max_restart = max_restart
        self.parameters['restart_interval'] = restart_interval
        self.parameters['restart_fraction'] = restart_fraction
        self.parameters['max_restart'] = max_restart
        return

    def _compute(self, base_lr: float, epoch: int) -> float:
        if epoch >= self.restart_interval:
            n_restart, restarted_epoch = divmod(epoch, self.restart_interval)
            if self.max_restart is None or n_restart <= self.max_restart:
                if restarted_epoch:
                    base_lr *= self.restart_fraction
                    epoch = restarted_epoch
                else:
                    epoch = self.restart_interval

        return self.logic(base_lr, epoch)


class WarmupScheduler(TransformScheduler):
    """Scheduler adding warmup to existing logic.

    Attributes:
        logic: callable to calculate the scheduling.
        warmup_steps: number of warmup steps.
        parameters: metadata associated with the scheduler.
        base_scheduler_name: name of the base scheduler for representation.
    """

    def __init__(self, logic: _SchedulingLogic, warmup_steps: int):
        """Initialize.

        Args:
            logic: callable to calculate the scheduling.
            warmup_steps: number of warmup steps.
        """
        super().__init__(logic)
        if warmup_steps < 0:
            raise ValueError('warmup_steps must be non-negative.')

        self.warmup_steps = warmup_steps
        self.parameters['warmup_steps'] = warmup_steps
        return

    def _compute(self, base_lr: float, epoch: int) -> float:
        if epoch < self.warmup_steps:
            return base_lr * (epoch / self.warmup_steps)

        return self.logic(base_lr, epoch - self.warmup_steps)


@dataclasses.dataclass(frozen=True)
class BaseScheduler(AbstractScheduler, abc.ABC):
    """Base class for schedulers that use dataclasses."""

    @property
    def base_scheduler_name(self) -> str:
        """Name of the base scheduler for representation."""
        return self.__class__.__name__

    @property
    def parameters(self) -> dict[str, Any]:
        """Metadata associated with the scheduler."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ConstantScheduler(BaseScheduler):
    """Constant learning rate."""

    def _compute(self, base_lr: float, epoch: int) -> float:
        return base_lr


@dataclasses.dataclass(frozen=True)
class PolynomialScheduler(BaseScheduler):
    """Polynomial learning rate scheduler: f(x) = C0 + C1(1 - x/C2)^C3.

    C0, C1, C2, C3 are defined so that:
    - f(x) = base_value when epoch = 0,
    - f(x) = min value when epoch is C2 = number of decay steps
    - f(x) is a polynomial of degree C3.

    After the number of decay steps, returns min value.

    Attributes:
        max_epochs: maximum number of epochs.
        power: polynomial power.
        min_decay: minimum fraction of the initial learning rate.
    """

    max_epochs: int = 1000
    power: float = 1.0
    min_decay: float = 0.0

    def __post_init__(self):
        """Input Validation."""
        if self.max_epochs <= 0:
            raise ValueError('max_epochs must be positive.')

        if self.power < 0:
            raise ValueError('power must be non-negative.')

        if not 0 <= self.min_decay <= 1:
            raise ValueError('min_decay must be between 0 and 1.')

        return

    @override
    def _compute(self, base_lr: float, epoch: int) -> float:
        if epoch >= self.max_epochs:
            return self.min_decay * base_lr

        decay_factor = (1 - epoch / self.max_epochs) ** self.power
        return self.min_decay + decay_factor * (1 - self.min_decay)


@dataclasses.dataclass(frozen=True)
class ExponentialScheduler(BaseScheduler):
    """Schedule exponential decay: f(x) = C0 + C1(C2^x).

    C0, C1, and C2 are defined so that:
    - f(x) = base_value when epoch = 0,
    - f(x) = min value when the epoch goes to infinite
    - f(x) is an exponential function with decay factor C2.

    After the number of decay steps, returns min value.

    Attributes:
        exp_decay: exponential decay parameter d for the curve: f(x) = Cd^x.
        min_decay: proportion of base learning rate for the minimum CO.
    """

    exp_decay: float = 0.975
    min_decay: float = 0.00

    def __post_init__(self):
        """Input Validation."""
        if not 0 < self.exp_decay <= 1:
            raise ValueError('exp_decay must be positive and less than 1.')

        if not 0 <= self.min_decay <= 1:
            raise ValueError('min_decay must be between 0 and 1.')

        return

    @override
    def _compute(self, base_lr: float, epoch: int) -> float:
        min_value = self.min_decay * base_lr
        return (base_lr - min_value) * self.exp_decay**epoch + min_value


@dataclasses.dataclass(frozen=True)
class CosineScheduler(BaseScheduler):
    """Schedule cosine decay: f(x) = C0 + C1(1 + cos(πx/C2)).

    C0, C1, and C2 are defined so that:
    - f(x) = base_value when epoch = 0 and,
    - f(x) = min value when epoch is C2 = number of decay steps.

    After the number of decay steps, returns min value.

    Attributes:
        decay_steps: number of steps (epochs) to reach maximum decay.
        min_decay: fraction of base_value for the minimum value.
    """

    decay_steps: int = 250
    min_decay: float = 0.01

    def __post_init__(self):
        """Input Validation."""
        if self.decay_steps <= 0:
            raise ValueError('decay_steps must be positive.')

        if not 0 <= self.min_decay <= 1:
            raise ValueError('min_decay must be between 0 and 1.')

        return

    @override
    def _compute(self, base_lr: float, epoch: int) -> float:
        min_lr = self.min_decay * base_lr
        if epoch > self.decay_steps:
            return min_lr

        from_1_to_minus1 = np.cos(np.pi * epoch / self.decay_steps)
        return min_lr + (base_lr - min_lr) * (1 + from_1_to_minus1) / 2


@dataclasses.dataclass(frozen=True)
class StepScheduler(BaseScheduler):
    """Step-wise learning rate scheduler.

    Reduces learning rate by a factor at specified milestones.

    Attributes:
        milestones: iterable of epochs at which to reduce the learning rate.
        gamma: factor by which to reduce learning rate.
    """

    milestones: Iterable[int] = dataclasses.field(default_factory=lambda: [200])
    gamma: float = 0.1

    def __post_init__(self):
        """Input Validation."""
        if not all(m > 0 for m in self.milestones):
            raise ValueError('All milestones must be positive.')

        if self.milestones != sorted(self.milestones):
            raise ValueError('Milestones must be in ascending order.')

        if not 0 < self.gamma <= 1:
            raise ValueError('gamma must be between 0 and 1 (exclusive of 0).')

        return

    @override
    def _compute(self, base_lr: float, epoch: int) -> float:
        count = sum(1 for milestone in self.milestones if epoch >= milestone)
        return base_lr * (self.gamma**count)


def rescale(factor: float) -> Callable[[_SchedulingLogic], RescaleScheduler]:
    """Create a scaling transformation.

    Args:
        factor: factor that rescales the value.

    Returns:
        A decorator that adds scaling to the scheduling logic.
    """

    def _decorator(logic: _SchedulingLogic) -> RescaleScheduler:
        return RescaleScheduler(logic, factor)

    return _decorator


def restart(
    restart_interval: int,
    restart_fraction: float = 1.0,
    max_restart: int | None = None,
) -> Callable[[_SchedulingLogic], RestartScheduler]:
    """Create a restart transformation.

    Args:
        restart_interval: number of epochs between restarts.
        restart_fraction: fraction to use when restarting.
        max_restart: Maximum number of restarts before deactivating.

    Returns:
        A decorator that adds restarting to the scheduling logic.
    """

    def _decorator(logic: _SchedulingLogic) -> RestartScheduler:
        return RestartScheduler(
            logic, restart_interval, restart_fraction, max_restart
        )

    return _decorator


def warmup(
    warmup_steps: int = 10,
) -> Callable[[_SchedulingLogic], WarmupScheduler]:
    """Create a warmup transformation.

    Args:
        warmup_steps: number of warmup steps.

    Returns:
        A decorator that adds warmup to the scheduling logic.
    """

    def _decorator(logic: _SchedulingLogic) -> WarmupScheduler:
        return WarmupScheduler(logic, warmup_steps)

    return _decorator

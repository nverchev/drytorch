"""Module containing schedulers for the learning rates."""

from __future__ import annotations

import abc
import dataclasses

from collections.abc import Callable, Iterable

import numpy as np

from typing_extensions import override

from drytorch.core import protocols as p


class AbstractScheduler(p.SchedulerProtocol, abc.ABC):
    """Abstract class for the scheduler."""

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
        f: Callable[[AbstractScheduler], AbstractScheduler],
        /,
    ) -> AbstractScheduler:
        """Allow transformation of the scheduler.

        Args:
            f: a function specifying the transformation.

        Returns:
            the transformed scheduler.
        """
        return f(self)

    @abc.abstractmethod
    def _compute(self, start_value: float, epoch: int) -> float:
        """Compute the scheduled value.

        Args:
            start_value: value when epoch is 0.
            epoch: variable of the function.

        Returns:
            the value for learning rate to use.
        """


# Need frozen=True to have it as the default value
@dataclasses.dataclass(frozen=True)
class ConstantScheduler(AbstractScheduler):
    """Constant learning rate."""

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        return start_value


@dataclasses.dataclass(frozen=True)
class PolynomialScheduler(AbstractScheduler):
    """Polynomial learning rate scheduler: f(x) = C0 + C1(1 - x/C2)^C3.

    C0, C1, C2, C3 are defined so that:
        f(x) = base_value when epoch = 0,
        f(x) = min value when epoch is C2 = number of decay steps and,
        f(x) is a polynomial of degree C3.
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

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        if epoch >= self.max_epochs:
            return self.min_decay * start_value

        decay_factor = (1 - epoch / self.max_epochs) ** self.power
        return self.min_decay + decay_factor * (1 - self.min_decay)


@dataclasses.dataclass(frozen=True)
class ExponentialScheduler(AbstractScheduler):
    """Schedule exponential decay: f(x) = C0 + C1(C2^x).

    C0, C1 and C2 are defined so that:
        f(x) = base_value when epoch = 0,
        f(x) = min value when the epoch goes to infinite and,
        f(x) is an exponential function with decay factor C2.
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

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        min_value = self.min_decay * start_value
        return (start_value - min_value) * self.exp_decay**epoch + min_value


@dataclasses.dataclass(frozen=True)
class CosineScheduler(AbstractScheduler):
    """Schedule cosine decay: f(x) = C0 + C1(1 + cos(πx/C2)).

    C0, C1 and C2 are defined so that:
        f(x) = base_value when epoch = 0 and,
        f(x) = min value when epoch is C2 = number of decay steps.
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

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        min_lr = self.min_decay * start_value
        if epoch > self.decay_steps:
            return min_lr

        from_1_to_minus1 = np.cos(np.pi * epoch / self.decay_steps)
        return min_lr + (start_value - min_lr) * (1 + from_1_to_minus1) / 2


@dataclasses.dataclass(frozen=True)
class RescaleScheduler(AbstractScheduler):
    """Scale the output of an existing scheduler.

    Attributes:
        factor: factor that rescales the value.
        base_scheduler: the scheduler to call.
    """

    base_scheduler: p.SchedulerProtocol
    factor: float

    def __post_init__(self):
        """Input Validation."""
        if self.factor <= 0:
            raise ValueError('factor must be positive.')

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        return self.factor * self.base_scheduler(start_value, epoch)


@dataclasses.dataclass(frozen=True)
class RestartScheduler(AbstractScheduler):
    """Wraps another scheduler to provide periodic restarts.

    Attributes:
        base_scheduler: the scheduler to restart.
        restart_interval: the number of epochs between restarts.
        restart_fraction: fraction of the base value to use as the base value
            when restarting.
        max_restart: Maximum number of restarts before deactivating. Default
            never deactivates.
    """

    base_scheduler: p.SchedulerProtocol
    restart_interval: int
    restart_fraction: float = 1.0
    max_restart: int | None = None

    def __post_init__(self):
        """Input Validation."""
        if self.restart_interval <= 0:
            raise ValueError('restart_interval must be positive.')

        if self.restart_fraction <= 0:
            raise ValueError('restart_fraction must be positive.')

        if self.max_restart is not None and self.max_restart <= 0:
            raise ValueError('max_restart must be positive.')

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        if epoch >= self.restart_interval:
            num_restart, restarted_epoch = divmod(epoch, self.restart_interval)
            if self.max_restart is None or num_restart <= self.max_restart:
                if restarted_epoch:
                    start_value *= self.restart_fraction
                    epoch = restarted_epoch
                else:
                    epoch = self.restart_interval

        return self.base_scheduler(start_value, epoch)


@dataclasses.dataclass(frozen=True)
class WarmupScheduler(AbstractScheduler):
    """Adds a warmup phase to any scheduler.

    During warmup, the learning rate increases linearly from 0 to base_lr.
    After warmup, delegates to the wrapped scheduler with adjusted epochs.

    Attributes:
        warmup_steps: number of steps (epochs) for the linear warmup phase.
        base_scheduler: the base scheduler to wrap with warmup.
    """

    base_scheduler: p.SchedulerProtocol
    warmup_steps: int = 10

    def __post_init__(self):
        """Input Validation."""
        if self.warmup_steps < 0:
            raise ValueError('warmup_steps must be non-negative.')

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        if epoch < self.warmup_steps:
            return start_value * (epoch / self.warmup_steps)

        return self.base_scheduler(start_value, epoch - self.warmup_steps)

    @override
    def __repr__(self) -> str:
        wrapped_repr = self.base_scheduler.__repr__()
        return f'{wrapped_repr} with {self.warmup_steps} warm-up steps'


@dataclasses.dataclass(frozen=True)
class StepScheduler(AbstractScheduler):
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

    @override
    def _compute(self, start_value: float, epoch: int) -> float:
        count = sum(1 for milestone in self.milestones if epoch >= milestone)
        return start_value * (self.gamma**count)


# Decorator functions for functional composition
def rescale(factor: float) -> Callable[[AbstractScheduler], AbstractScheduler]:
    """Create a scaling transformation.

    Args:
        factor: factor that rescales the value.

    Returns:
        A decorator that wraps a scheduler with scaling.
    """

    def _decorator(scheduler: AbstractScheduler) -> AbstractScheduler:
        return RescaleScheduler(scheduler, factor)

    return _decorator


def restart(
    restart_interval: int,
    restart_fraction: float = 1.0,
) -> Callable[[AbstractScheduler], AbstractScheduler]:
    """Create a restart transformation.

    Args:
        restart_interval: number of epochs between restarts.
        restart_fraction: fraction to use when restarting.

    Returns:
        A decorator that wraps a scheduler with restarts.
    """

    def _decorator(scheduler: AbstractScheduler) -> AbstractScheduler:
        return RestartScheduler(scheduler, restart_interval, restart_fraction)

    return _decorator


def warmup(
    warmup_steps: int = 10,
) -> Callable[[AbstractScheduler], AbstractScheduler]:
    """Create a warmup transformation.

    Args:
        warmup_steps: number of warmup steps.

    Returns:
        A decorator that wraps a scheduler with warmup.
    """

    def _decorator(scheduler: AbstractScheduler) -> AbstractScheduler:
        return WarmupScheduler(scheduler, warmup_steps)

    return _decorator

"""This module defines schedulers for the learning rates."""
from abc import abstractmethod
import dataclasses

import numpy as np

from src.dry_torch import protocols as p


class AbstractScheduler(p.SchedulerProtocol):
    """Abstract class for the scheduler."""

    def __call__(self, base_lr: float, epoch: int) -> float:
        """
        Modifies the learning rate according to a schedule.

        Args:
            base_lr: initial learning rate.
            epoch: the current epoch.
        Returns:
            scheduled value for the learning rate.
        """
        return self._compute(base_lr, epoch)

    @abstractmethod
    def _compute(self, base_lr: float, epoch: int) -> float:
        ...


@dataclasses.dataclass(frozen=True)
class ConstantScheduler(AbstractScheduler):
    """
    Constant learning rate.

    Attributes:
        factor: multiplicative factor for the base learning rate.
    """
    factor: float = 1.0

    def _compute(self, base_lr: float, epoch: int) -> float:
        return self.factor * base_lr


@dataclasses.dataclass
class ExponentialScheduler(AbstractScheduler):
    """
    Schedule following an exponential decay.

    Attributes:
        exp_decay: exponential decay parameter d for the curve f(x) = Ce^(dx).
    """
    exp_decay: float = .975
    min_decay: float = 0.00

    def _compute(self, base_lr: float, epoch: int) -> float:
        return max(base_lr * self.exp_decay ** epoch, self.min_decay)


@dataclasses.dataclass
class CosineScheduler(AbstractScheduler):
    """
    Learning rate with cosine decay.

    The cosine function is f(x) = C0 + C(1 + cos(C2x)) specified by the
    following parameters. It remains constant after reaching the minimum value.

    Attributes:
        decay_steps: the epochs (C2 * pi) were the schedule follows a cosine
            curve until its minimum C0.
        min_decay: the fraction of the initial value that it returned at the
            end (C0 + C) / C0.
    """

    decay_steps: int = 250
    min_decay: float = 0.01

    def _compute(self, base_lr: float, epoch: int) -> float:
        min_lr = self.min_decay * base_lr
        if epoch >= self.decay_steps:
            return min_lr
        from_1_to_minus1 = np.cos(np.pi * epoch / self.decay_steps)
        return min_lr + (base_lr - min_lr) * (1 + from_1_to_minus1) / 2


@dataclasses.dataclass
class WarmupScheduler(AbstractScheduler):
    """
    Adds a warmup phase to any scheduler.

    During warmup, the learning rate increases linearly from 0 to base_lr.
    After warmup, delegates to the wrapped scheduler with adjusted epochs.

    Attributes:
        warmup_steps: Number of epochs for the linear warmup phase.
        scheduler: The base scheduler to wrap with warmup.
    """

    warmup_steps: int = 10
    scheduler: p.SchedulerProtocol = ConstantScheduler()

    def _compute(self, base_lr: float, epoch: int) -> float:
        if epoch < self.warmup_steps:
            return base_lr * (epoch / self.warmup_steps)
        return self.scheduler(base_lr, epoch - self.warmup_steps)

    def __repr__(self) -> str:
        wrapped_repr = self.scheduler.__repr__()
        return f'{wrapped_repr} with {self.warmup_steps} warm up steps'


@dataclasses.dataclass
class CompositionScheduler(AbstractScheduler):
    """
    Compose two schedulers as in function composition.

    Attributes:
        scheduler_a: second scheduler to call.
        scheduler_b: first scheduler to call.
    """

    scheduler_a: p.SchedulerProtocol
    scheduler_b: p.SchedulerProtocol

    def _compute(self, base_lr: float, epoch: int) -> float:
        return self.scheduler_a(self.scheduler_b(base_lr, epoch), epoch)

"""Module containing schedulers for the learning rates."""

from abc import abstractmethod
import dataclasses

import numpy as np

from drytorch import protocols as p


class AbstractScheduler(p.SchedulerProtocol):
    """Abstract class for the scheduler."""

    def __call__(self, base_lr: float, epoch: int) -> float:
        """
        Modify the learning rate according to a schedule.

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

    @abstractmethod
    def _compute(self, start_value: float, epoch: int) -> float:
        """
        Compute the scheduled value.

        Args:
            start_value: value when epoch is 0.
            epoch: variable of the function.
        Return:
            the value for learning rate to use.
        """


# Need frozen=True to have it as default value
@dataclasses.dataclass(frozen=True)
class ConstantScheduler(AbstractScheduler):
    """Constant learning rate."""

    def _compute(self, start_value: float, epoch: int) -> float:
        return start_value


@dataclasses.dataclass
class LinearScheduler(AbstractScheduler):
    """
    Schedule linear decay: f(x) = C0 + C1*x.

    C0 and C1 are defined so that:
        f(x) = base_value when epoch = 0 and,
        f(x) = min value when epoch is C2 = number of decay steps.
    After the number of decay steps, returns min value.

    Attributes:
        decay_steps: number of steps (epochs) to reach maximum decay.
        min_decay: fraction of base_value for the minimum value.
    """
    decay_steps: int = 250
    min_decay: float = 0.01

    def _compute(self, start_value: float, epoch: int) -> float:
        min_lr = self.min_decay * start_value
        if epoch < self.decay_steps:
            # Linear decay from start_value to min_lr
            decay_rate = 1 - (epoch / self.decay_steps)
            return min_lr + (start_value - min_lr) * decay_rate
        else:
            return min_lr


@dataclasses.dataclass
class ExponentialScheduler(AbstractScheduler):
    """
    Schedule exponential decay: f(x) = Cd^x.

    C is the base learning rate. Return value has a minimum value C0.

    Attributes:
        exp_decay: exponential decay parameter d for the curve: f(x) = Cd^x.
        min_decay: proportion of base learning rate for the minimum CO.
    """
    exp_decay: float = .975
    min_decay: float = 0.00

    def _compute(self, start_value: float, epoch: int) -> float:
        min_value = self.min_decay * start_value
        return max(start_value * self.exp_decay ** epoch, min_value)


@dataclasses.dataclass
class CosineScheduler(AbstractScheduler):
    """
    Schedule cosine decay: f(x) = C0 + C1(1 + cos(πx/C2)).

    C0 and C1 are defined so that:
        f(x) = base_value when epoch = 0 and,
        f(x) = min value when epoch is C2 = number of decay steps.
    After the number of decay steps, returns min value.

    Attributes:
        decay_steps: number of steps (epochs) to reach maximum decay.
        min_decay: fraction of base_value for the minimum value.
        restart: whether to restart the annealing every decay_steps epochs.
        restart_factor: factor of base learning rate value at restart.
    """

    decay_steps: int = 250
    min_decay: float = 0.01
    restart: bool = False
    restart_factor: float = 1.

    def _compute(self, start_value: float, epoch: int) -> float:
        min_lr = self.min_decay * start_value
        if epoch > self.decay_steps:
            if self.restart:
                epoch = epoch % self.decay_steps or self.decay_steps
            else:
                return min_lr
        from_1_to_minus1 = np.cos(np.pi * epoch / self.decay_steps)
        return min_lr + (start_value - min_lr) * (1 + from_1_to_minus1) / 2


@dataclasses.dataclass
class WarmupScheduler(AbstractScheduler):
    """
    Adds a warmup phase to any scheduler.

    During warmup, the learning rate increases linearly from 0 to base_lr.
    After warmup, delegates to the wrapped scheduler with adjusted epochs.

    Attributes:
        warmup_steps: number of steps (epochs) for the linear warmup phase.
        scheduler: the base scheduler to wrap with warmup.
    """

    warmup_steps: int = 10
    scheduler: p.SchedulerProtocol = ConstantScheduler()

    def _compute(self, start_value: float, epoch: int) -> float:
        if epoch < self.warmup_steps:
            return start_value * (epoch / self.warmup_steps)

        return self.scheduler(start_value, epoch - self.warmup_steps)

    def __repr__(self) -> str:
        wrapped_repr = self.scheduler.__repr__()
        return f'{wrapped_repr} with {self.warmup_steps} warm up steps'


@dataclasses.dataclass
class FactorScheduler(AbstractScheduler):
    """
    Modify start value to an existing scheduler.

    Attributes:
        factor: factor for the start value of the scheduler.
        scheduler: the scheduler to call.
    """

    factor: float
    scheduler: p.SchedulerProtocol

    def _compute(self, start_value: float, epoch: int) -> float:
        return self.scheduler(self.factor * start_value, epoch)

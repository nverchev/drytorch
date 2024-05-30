from abc import abstractmethod
import numpy as np

from dry_torch import protocols


class AbstractScheduler(protocols.SchedulerProtocol):

    def __call__(self, base_lr: float, epoch: int) -> float:
        """
           Call the scheduler for the decay of the learning rate.

           Args:
               base_lr: initial learning rate when epoch is 0.
               epoch: the variable of decaying curve.
           Returns:
               decayed value for the learning rate.
        """
        return self._compute(base_lr, epoch)

    @abstractmethod
    def _compute(self, base_lr: float, epoch: int) -> float:
        ...


class ConstantScheduler(AbstractScheduler):
    """
    Constant learning rate.
    """

    def _compute(self, base_lr: float, epoch: int) -> float:
        return base_lr

    def __repr__(self) -> str:
        return 'Constant learning rate.'


class ExponentialScheduler(AbstractScheduler):
    """
    Learning rate with exponential decay.
    Args:
        exp_decay: exponential decay parameter d for the curve f(x) = Ce^(dx).
    """

    def __init__(self, exp_decay: float = .975) -> None:
        self.exp_decay = exp_decay

    def _compute(self, base_lr: float, epoch: int) -> float:
        return base_lr * self.exp_decay ** epoch

    def __repr__(self) -> str:
        desc = 'Exponential schedule with exponential decay = {}.'
        return desc.format(self.exp_decay)


class CosineScheduler(AbstractScheduler):
    """
    Learning rate with cosine decay.
    It remains constant after reaching the minimum value.
    The cosine function is f(x) = C0 + C(1 + cos(C2x))
    specified by the following parameters.

    Args:
        decay_steps: the epochs (C2 * pi) were the schedule follows a cosine
         curve until its minimum C0.
        min_decay: the fraction of the initial value that it returned at the
            end (C0 + C) / C0.
    """

    def __init__(self, decay_steps: int = 250, min_decay: float = 0.01):
        self.decay_steps = decay_steps
        self.min_decay = min_decay

    def _compute(self, base_lr: float, epoch: int) -> float:
        min_lr = self.min_decay * base_lr
        if epoch >= self.decay_steps:
            return min_lr
        from_1_to_minus1 = np.cos(np.pi * epoch / self.decay_steps)
        return min_lr + (base_lr - min_lr) * (1 + from_1_to_minus1) / 2

    def __repr__(self) -> str:
        desc = 'Cosine schedule with {} decay steps and {} min_decay factor.'
        return desc.format(self.decay_steps, self.min_decay)

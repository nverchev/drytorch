"""Module containing classes with learning algorithm's specifications."""

from __future__ import annotations

import dataclasses

from typing import Any

import torch

from drytorch.core import protocols as p
from drytorch.lib import gradient_ops, schedulers


_default_scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
_default_grad_op: p.GradientOpProtocol = gradient_ops.NoOp()


@dataclasses.dataclass(frozen=True)
class LearningSchema(p.LearningProtocol):
    """Class with specifications for the learning algorithm.

    Attributes:
        optimizer_cls: the optimizer class to bind to the module.
        base_lr: initial learning rates for named parameters or global value.
        optimizer_defaults: optional arguments for the optimizer.
        scheduler: modifies the learning rate given the current epoch.
        gradient_op: modifies parameters' after backward propagation.
    """

    optimizer_cls: type[torch.optim.Optimizer]
    base_lr: float | dict[str, float]
    scheduler: p.SchedulerProtocol = _default_scheduler
    optimizer_defaults: dict[str, Any] = dataclasses.field(default_factory=dict)
    gradient_op: p.GradientOpProtocol = _default_grad_op

    @classmethod
    def adam(
        cls,
        base_lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        scheduler: p.SchedulerProtocol = _default_scheduler,
        gradient_op: p.GradientOpProtocol = _default_grad_op,
    ) -> LearningSchema:
        """Convenience method for the Adam optimizer.

        Args:
            base_lr: initial learning rate.
            betas: coefficients used for computing running averages.
            scheduler: modifies the learning rate given the current epoch.
            gradient_op: modifies parameters' after backward propagation.
        """
        return cls(
            optimizer_cls=torch.optim.Adam,
            base_lr=base_lr,
            scheduler=scheduler,
            optimizer_defaults={'betas': betas},
            gradient_op=gradient_op,
        )

    @classmethod
    def adam_w(
        cls,
        base_lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        scheduler: p.SchedulerProtocol = _default_scheduler,
        gradient_op: p.GradientOpProtocol = _default_grad_op,
    ) -> LearningSchema:
        """Convenience method for the AdamW optimizer.

        Args:
            base_lr: initial learning rate.
            betas: coefficients used for computing running averages.
            weight_decay: weight decay (L2 penalty).
            scheduler: modifies the learning rate given the current epoch.
            gradient_op: modifies parameters' after backward propagation.
        """
        return cls(
            optimizer_cls=torch.optim.AdamW,
            base_lr=base_lr,
            scheduler=scheduler,
            optimizer_defaults={'betas': betas, 'weight_decay': weight_decay},
            gradient_op=gradient_op,
        )

    @classmethod
    def sgd(
        cls,
        base_lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        scheduler: p.SchedulerProtocol = _default_scheduler,
        gradient_op: p.GradientOpProtocol = _default_grad_op,
    ) -> LearningSchema:
        """Convenience method for the SGD optimizer.

        Args:
            base_lr: initial learning rate.
            momentum: momentum factor.
            dampening:  dampening for momentum.
            weight_decay: weight decay (L2 penalty).
            nesterov: enables Nesterov momentum.
            scheduler: modifies the learning rate given the current epoch.
            gradient_op: modifies parameters' after backward propagation.
        """
        return cls(
            optimizer_cls=torch.optim.SGD,
            base_lr=base_lr,
            scheduler=scheduler,
            optimizer_defaults={
                'momentum': momentum,
                'weight_decay': weight_decay,
                'dampening': dampening,
                'nesterov': nesterov,
            },
            gradient_op=gradient_op,
        )

    @classmethod
    def r_adam(
        cls,
        base_lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        scheduler: p.SchedulerProtocol = _default_scheduler,
        gradient_op: p.GradientOpProtocol = _default_grad_op,
    ) -> LearningSchema:
        """Convenience method for the RAdam optimizer.

        Args:
            base_lr: initial learning rate.
            betas: coefficients used for computing running averages.
            weight_decay: weight decay (L2 penalty).
            scheduler: modifies the learning rate given the current epoch.
            gradient_op: modifies parameters' after backward propagation.
        """
        wd_flag = bool(weight_decay)
        return cls(
            optimizer_cls=torch.optim.RAdam,
            base_lr=base_lr,
            scheduler=scheduler,
            optimizer_defaults={
                'betas': betas,
                'weight_decay': weight_decay,
                'decoupled_weight_decay': wd_flag,
            },
            gradient_op=gradient_op,
        )

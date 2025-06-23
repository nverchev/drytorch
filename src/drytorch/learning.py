"""Module containing classes with learning algorithm's specifications."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

import torch

from drytorch import protocols as p
from drytorch import schedulers


@dataclasses.dataclass
class LearningScheme(p.LearningProtocol):
    """
    Class with specifications for the learning algorithm.

    Attributes:
        optimizer_cls: the optimizer class to bind to the module.
        base_lr: initial learning rates for named parameters or global value.
        optimizer_defaults: optional arguments for the optimizer.
        scheduler: modifies the learning rate given the current epoch.
        clip_strategy: strategy to clip gradients. See utils.gradient_clipping.
    """
    optimizer_cls: type[torch.optim.Optimizer]
    base_lr: float | dict[str, float]
    scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
    optimizer_defaults: dict[str, Any] = dataclasses.field(default_factory=dict)
    clip_strategy: Callable = dataclasses.field(default=lambda x: None)

    @classmethod
    def Adam(cls,
             base_lr: float = 1e-3,
             betas: tuple[float, float] = (0.9, 0.999),
             scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
             ) -> LearningScheme:
        """
        Convenience method for the Adam optimizer.

        Args:
            base_lr: initial learning rate.
            betas: coefficients used for computing running averages.
            scheduler: modifies the learning rate given the current epoch.
        """
        return cls(optimizer_cls=torch.optim.Adam,
                   base_lr=base_lr,
                   scheduler=scheduler,
                   optimizer_defaults={'betas': betas})

    @classmethod
    def AdamW(cls,
              base_lr: float = 1e-3,
              betas: tuple[float, float] = (0.9, 0.999),
              weight_decay: float = 1e-2,
              scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
              ) -> LearningScheme:
        """
        Convenience method for the AdamW optimizer.

        Args:
            base_lr: initial learning rate.
            betas: coefficients used for computing running averages.
            weight_decay: weight decay (L2 penalty).
            scheduler: modifies the learning rate given the current epoch.
        """
        return cls(optimizer_cls=torch.optim.AdamW,
                   base_lr=base_lr,
                   scheduler=scheduler,
                   optimizer_defaults={'betas': betas,
                                       'weight_decay': weight_decay})

    @classmethod
    def SGD(cls,
            base_lr: float = 0.01,
            momentum: float = 0.,
            weight_decay: float = 0.,
            dampening: float = 0.,
            nesterov: bool = False,
            scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
            ) -> LearningScheme:
        """
        Convenience method for the SGD optimizer.

        Args:
            base_lr: initial learning rate.
            momentum: momentum factor.
            dampening:  dampening for momentum.
            weight_decay: weight decay (L2 penalty).
            nesterov: enables Nesterov momentum.
            scheduler: modifies the learning rate given the current epoch.
        """
        return cls(optimizer_cls=torch.optim.SGD,
                   base_lr=base_lr,
                   scheduler=scheduler,
                   optimizer_defaults={'momentum': momentum,
                                       'weight_decay': weight_decay,
                                       'dampening': dampening,
                                       'nesterov': nesterov})

    @classmethod
    def RAdam(cls,
              base_lr: float = 1e-3,
              betas: tuple[float, float] = (0.9, 0.999),
              weight_decay: float = 0.,
              scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
              ) -> LearningScheme:
        """
        Convenience method for the RAdam optimizer.

        Args:
            base_lr: initial learning rate.
            betas: coefficients used for computing running averages.
            weight_decay: weight decay (L2 penalty).
            scheduler: modifies the learning rate given the current epoch.
        """
        wd_flag = bool(weight_decay)
        return cls(optimizer_cls=torch.optim.RAdam,
                   base_lr=base_lr,
                   scheduler=scheduler,
                   optimizer_defaults={'betas': betas,
                                       'weight_decay': weight_decay,
                                       'decoupled_weight_decay': wd_flag})

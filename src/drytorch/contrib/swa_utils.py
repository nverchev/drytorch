"""Utilities for Stochastic Weight Averaging."""

from typing import TypeVar

import torch

from drytorch.core import protocols as p
from drytorch.lib import runners


__all__ = [
    'ModelMomentaUpdater',
]


Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)

AbstractBatchNorm = torch.nn.modules.batchnorm._BatchNorm


class ModelMomentaUpdater(runners.ModelRunner[Input, Target, Output]):
    """Update the momenta in the batch normalization layers."""

    def __call__(self, store_outputs: bool = False) -> None:
        """Single pass on the dataset."""
        momenta = dict[AbstractBatchNorm, float | None]()
        for module in self.model.module.modules():
            if isinstance(module, AbstractBatchNorm):
                module.reset_running_stats()
                momenta[module] = module.momentum

        if not momenta:
            return

        for module in momenta.keys():
            module.momentum = None

        was_training = self.model.module.training
        self.model.module.train()
        super().__call__(store_outputs)

        for bn_module in momenta:
            bn_module.momentum = momenta[bn_module]

        self.model.module.train(was_training)
        return

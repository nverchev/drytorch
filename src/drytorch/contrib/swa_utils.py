"""Utilities for Stochastic Weight Averaging."""

from typing import TypeVar

import torch

from drytorch.core import protocols as p
from drytorch.lib import runners


_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)

AbstractBatchNorm = torch.nn.modules.batchnorm._BatchNorm


class ModelMomentaUpdater(runners.ModelCaller[_Input, _Output]):
    """Update the momenta in the batch normalization layers."""

    def __call__(self) -> None:
        """Single pass on the dataset."""
        super().__call__()
        momenta = dict[AbstractBatchNorm, float | None]()
        for module in self.model.module.modules():
            if isinstance(module, AbstractBatchNorm):
                module.reset_running_stats()
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = self.model.module.training
        self.model.module.train()
        for module in momenta.keys():
            module.momentum = None

        for bn_module in momenta:
            bn_module.momentum = momenta[bn_module]

        self.model.module.train(was_training)
        return

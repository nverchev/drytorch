"""Utilities for Stochastic Weight Averaging"""

import torch

from drytorch.running import ModelCaller

AbstractBatchNorm = torch.nn.modules.batchnorm._BatchNorm


class ModelMomentaUpdater(ModelCaller):

    def __call__(self) -> None:
        """Single pass on the dataset."""
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

        super().__call__()
        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]

        self.model.module.train(was_training)
        return

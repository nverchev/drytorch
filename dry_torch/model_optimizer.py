from typing import Generic, Optional, Any

import torch

from dry_torch import Scheduler, ConstantScheduler
from dry_torch.protocols import TypedModule,  OptParams, ModuleInput, ModuleOutput


class ModelOptimizer(Generic[ModuleInput, ModuleOutput]):
    """
    Bundle the model and its optimizer.
    Support different learning rates and separate parameters groups.

    Args:
        model: Pytorch model to optimize. TypedModule type variables match the class type variables.
        optimizer_cls: the optimizer class to bind to the model. Defaults to torch.optim.Adam.
        lr: a dictionary of learning rates for the named parameters or a float for a global value.
        other_optimizer_args: optional arguments for the optimizer (same for all the parameters).
        ing to the epoch. Default value does not implement a scheduler.
        device: the device where to store the model. Default uses cuda when available else cpu.
        epoch: the current epoch, that is, the number of epochs the model has been trainer plus one. Defaults to 0.

    Attributes:
        model: Pytorch model to optimize. TypedModule type variables match the class type variables.
        optimizer: the optimizer bound to the model.
        device: the device where to store the model.
        epoch: the current epoch, that is, the number of epochs the model has been trainer plus one.

    Methods:
        params_lr: property for the updated learning rates for the optimizer.
        update_learning_rate(): update the optimizer with the updated settings or with an input learning rate.
    """

    def __init__(self, model: TypedModule[ModuleInput, ModuleOutput] | torch.nn.Module,
                 optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
                 lr: float | dict[str, float] = 0.001,
                 other_optimizer_args: Optional[dict[str, Any]] = None,
                 scheduler: Scheduler = ConstantScheduler(),
                 device: Optional[torch.device] = None,
                 epoch: int = 0,
                 ) -> None:
        default_device: torch.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device: torch.device = default_device if device is None else device
        self.model: TypedModule[ModuleInput, ModuleOutput] | torch.nn.Module = model.to(device)
        self.params_lr = lr
        self.scheduler: Scheduler = scheduler
        self.epoch: int = epoch
        optimizer_args: dict[str, Any] = {} if other_optimizer_args is None else other_optimizer_args
        self.optimizer: torch.optim.Optimizer = optimizer_cls(params=self.params_lr, **optimizer_args)

    @property
    def params_lr(self) -> list[OptParams]:  # metadata depend on the epoch
        """
        Learning rates for each parameter updated according to the scheduler and the current epoch.
        """
        return [OptParams(params=g['params'], lr=self.scheduler(g['lr'], self.epoch)) for g in self._params_lr]

    @params_lr.setter
    def params_lr(self, lr: float | dict[str, float]) -> None:
        """
        Pair the learning rates with the parameter groups.

        Args:
            lr: a dictionary of learning rates for the named parameters or a float for a global value.
        """
        if isinstance(lr, float):
            self._params_lr = [OptParams(params=self.model.parameters(), lr=lr)]
        else:
            self._params_lr = [OptParams(params=getattr(self.model, k).parameters(), lr=v) for k, v in lr.items()]
            if sum(map(lambda x: len(list(x['params'])), self._params_lr)) < len(list(self.model.parameters())):
                raise ValueError(f'Input {lr} does not include for all the parameters in the model.')
        return

    def update_learning_rate(self, lr: Optional[float | dict[str, float]] = None) -> None:
        """
        It updates the learning rates for each parameters' group in the optimizer.

        Args:
            lr: a dictionary of learning rates for the named parameters or a float for a global value.
                If None (default), the scheduled learning rates are used.
        """
        if lr is not None:
            self.params_lr = lr
        for g, up_g in zip(self.optimizer.param_groups, self.params_lr):
            g['lr'] = up_g['lr']
        return

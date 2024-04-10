from typing import Generic, Any, Optional

import torch

from custom_trainer import Scheduler, ConstantScheduler
from custom_trainer.protocols import TypedModule, OptParams
from custom_trainer.trainer import TensorInputs, TensorOutputs


class ModelHandler(Generic[TensorInputs, TensorOutputs]):
    """
    Args:
            model: Pytorch model with a settings attribute which is a dictionary for extra details
            device: only gpu and cpu are supported
            optimizer_cls: a Pytorch optimizer class
            optim_args: arguments for the optimizer (see the optimizer_settings setter for more details)
    """

    def __init__(self, model: TypedModule[TensorInputs, TensorOutputs] | torch.nn.Module,
                 optimizer_cls: type[torch.optim.Optimizer],
                 optim_args: dict[str, Any],
                 scheduler: Scheduler = ConstantScheduler(),
                 device: Optional[torch.device] = None,
                 epoch: int = 0,
                 ) -> None:

        self.device: torch.device = torch.device('cuda:0') if device is None else device
        self.model: TypedModule[TensorInputs, TensorOutputs] | torch.nn.Module = model.to(device)
        self.optimizer_settings: dict[str, Any] = optim_args.copy()  # property that updates the learning rate
        self.optimizer: torch.optim.Optimizer = optimizer_cls(**self.optimizer_settings)
        self.scheduler: Scheduler = scheduler
        self.epoch: int = epoch

    @property
    def optimizer_settings(self) -> dict:  # settings depend on the epoch
        """
        It implements the scheduler and separate learning rates for the parameter groups.
        If the optimizer is correctly updated, it should be a copy of its params groups

        Returns:
            list of dictionaries with parameters and their updated learning rates plus the other fixed settings
        """
        optim_groups = self._optimizer_settings[0]
        params = [{'params': group['params'], 'lr': self.scheduler(group['lr'], self.epoch)} for group in optim_groups]
        return {'params': params, **self._optimizer_settings[1]}

    @optimizer_settings.setter
    def optimizer_settings(self, optim_args: dict[str, Any]) -> None:
        """
        It pairs the learning rates to the parameter groups.
        If only one number for the learning rate is given, it uses its value for each parameter

        Args:
            optim_args are similar to the default for the optimizer, a dictionary with a required lr key
            The only difference is that lr can be a dict of the form {parameter_name: learning_rate}
        """
        lr: float | dict[str, float] = optim_args.pop('lr')  # removes 'lr' as setting, we move it inside 'params'
        if isinstance(lr, dict):  # support individual lr for each parameter (for fine-tuning for example)
            params = [OptParams(params=getattr(self.model, k).parameters(), lr=v) for k, v in lr.items()]
            self._optimizer_settings: tuple[list[OptParams], dict[str, Any]] = params, optim_args
        else:
            self._optimizer_settings = [OptParams(params=self.model.parameters(), lr=lr)], optim_args
        return

    def update_learning_rate(self, new_lr: Optional[float | list[dict[str, float]]] = None) -> None:
        """
        It updates the learning rates of the optimizer.

        Args:
            new_lr: a global learning rate for all the parameters or a list of dict with a 'lr' as a key
            If you call it externally, make sure the list has the same order as in self.optimizer_settings
        """
        if isinstance(new_lr, float):  # transforms to list
            lr: list[dict[str, float]] = [{'lr': new_lr} for _ in self.optimizer.param_groups]
        elif new_lr is None:
            lr = self.optimizer_settings['params']
        else:
            lr = new_lr
        for g, up_g in zip(self.optimizer.param_groups, lr):
            g['lr'] = up_g['lr']
        return

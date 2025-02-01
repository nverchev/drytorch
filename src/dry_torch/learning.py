"""Classes for wrapping a torch module and its optimizer."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Optional, Self, TypedDict, TypeVar, cast
import copy

import torch
from torch import cuda
import dataclasses

from src.dry_torch import checkpoint
from src.dry_torch import exceptions
from src.dry_torch import protocols as p
from src.dry_torch import repr_utils
from src.dry_torch import registering
from src.dry_torch import schedulers
from src.dry_torch import tracking

_Input_contra = TypeVar('_Input_contra',
                        bound=p.InputType,
                        contravariant=True)

_Output_co = TypeVar('_Output_co',
                     bound=p.OutputType,
                     covariant=True)


class _OptParams(TypedDict):
    params: Iterator[torch.nn.Parameter]
    lr: float


@dataclasses.dataclass
class LearningScheme(p.LearningProtocol):
    """
    Class with specifications for the learning algorithm.

    Attributes:
        optimizer_cls: the optimizer class to bind to the module.
        lr: initial learning rates for the named parameters or global value.
        optimizer_defaults: optional arguments for the optimizer.
        scheduler: modifies the learning rate given the current epoch.
    """
    optimizer_cls: type[torch.optim.Optimizer]
    lr: float | dict[str, float]
    scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
    optimizer_defaults: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def Adam(cls,
             lr: float = 1e-3,
             betas: tuple[float, float] = (0.9, 0.999),
             scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
             ) -> LearningScheme:
        """
        Convenience method for the Adam optimizer.

        Args:
            lr: initial learning rate.
            betas: coefficients used for computing running averages.
            scheduler: modifies the learning rate given the current epoch.
        """
        return cls(optimizer_cls=torch.optim.Adam,
                   lr=lr,
                   scheduler=scheduler,
                   optimizer_defaults={'betas': betas})

    @classmethod
    def AdamW(cls,
              lr: float = 1e-3,
              betas: tuple[float, float] = (0.9, 0.999),
              weight_decay: float = 1e-2,
              scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
              ) -> LearningScheme:
        """
        Convenience method for the AdamW optimizer.

        Args:
            lr: initial learning rate.
            betas: coefficients used for computing running averages.
            weight_decay: weight decay (L2 penalty).
            scheduler: modifies the learning rate given the current epoch.
        """
        return cls(optimizer_cls=torch.optim.AdamW,
                   lr=lr,
                   scheduler=scheduler,
                   optimizer_defaults={'betas': betas,
                                       'weight_decay': weight_decay})

    @classmethod
    def SGD(cls,
            lr: float = 0.01,
            momentum: float = 0.,
            weight_decay: float = 0.,
            dampening: float = 0.,
            nesterov: bool = False,
            scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
            ) -> LearningScheme:
        """
        Convenience method for the SGD optimizer.

        Args:
            lr: initial learning rate.
            momentum: momentum factor.
            dampening:  dampening for momentum.
            weight_decay: weight decay (L2 penalty).
            nesterov: enables Nesterov momentum.
            scheduler: modifies the learning rate given the current epoch.
        """
        return cls(optimizer_cls=torch.optim.SGD,
                   lr=lr,
                   scheduler=scheduler,
                   optimizer_defaults={'momentum': momentum,
                                       'weight_decay': weight_decay,
                                       'dampening': dampening,
                                       'nesterov': nesterov})

    @classmethod
    def RAdam(cls,
              lr: float = 1e-3,
              betas: tuple[float, float] = (0.9, 0.999),
              weight_decay: float = 0.,
              scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
              ) -> LearningScheme:
        """
        Convenience method for the RAdam optimizer.

        Args:
            lr: initial learning rate.
            betas: coefficients used for computing running averages.
            weight_decay: weight decay (L2 penalty).
            scheduler: modifies the learning rate given the current epoch.
        """
        wd_flag = bool(weight_decay)
        return cls(optimizer_cls=torch.optim.RAdam,
                   lr=lr,
                   scheduler=scheduler,
                   optimizer_defaults={'betas': betas,
                                       'weight_decay': weight_decay,
                                       'decoupled_weight_decay': wd_flag})


class Model(p.ModelProtocol[_Input_contra, _Output_co]):
    """
    Wrapper for a torch.nn.Module class with extra information.

    Args:
        torch_module: Pytorch module with type annotations.
        name: the name of the model.
        device: the device where to store the weights of module. Default uses
            cuda when available else cpu.

    Attributes:
        module: Pytorch module to optimize.
        epoch: the number of epochs the model has been trained so far.

    """
    _default_name = repr_utils.DefaultName()

    def __init__(
            self,
            torch_module: p.ModuleProtocol[_Input_contra, _Output_co],
            /,
            name: Optional[str] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        self.module = self._validate_module(torch_module)
        self.name = repr_utils.StrWithTS(name or self._default_name)
        self.epoch: int = 0
        self.device = self._default_device() if device is None else device
        exp = registering.register_model(self)
        self._model_state_io = checkpoint.ModelStateIO(self, exp.dir)

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.module.to(device)
        return

    def __call__(self, inputs: _Input_contra) -> _Output_co:
        return self.module(inputs)

    def clone(self, new_name: str) -> Self:
        """
        Return a deepcopy of the object.
        """
        cloned_model = self._copy_module()
        cloned = self.__class__(
            cloned_model,
            name=new_name,
            device=self.device,
        )
        return cloned

    def increment_epoch(self) -> None:
        """Increment the epoch by 1."""
        self.epoch += 1

    def load_state(self, epoch=-1) -> None:
        """Load the weights and epoch of the model"""
        self._model_state_io.load(epoch=epoch)

    def save_state(self) -> None:
        """Save the weights and epoch of the model"""
        self._model_state_io.save()

    def to(self, device: torch.device) -> None:
        """Forward the homonymous method."""
        self.device = device

    def _copy_module(
            self
    ) -> p.ModuleProtocol[_Input_contra, _Output_co]:
        return copy.deepcopy(self.module)

    @staticmethod
    def _default_device() -> torch.device:
        return torch.device('cuda:0' if cuda.is_available() else 'cpu')

    @staticmethod
    def _validate_module(torch_model) -> torch.nn.Module:
        if not isinstance(torch_model, torch.nn.Module):
            raise TypeError('torch_module must be a torch.nn.Module subclass')
        return torch_model


class ModelOptimizer:
    """
    Bundle the module and its optimizer.
    Support different learning rates to separate parameters groups.

    Args:
        model: the model to be optimized.
        learning_scheme: the learning scheme for the optimizer.

    Attributes:
        model: the model to be optimized.
        module: the module contained in the model.
        scheduler: the scheduler for the learning rate.
        optimizer: the optimizer bound to the module.
    """

    def __init__(
            self,
            model: p.ModelProtocol[_Input_contra, _Output_co],
            learning_scheme: p.LearningProtocol
    ) -> None:
        self.model = model
        self.module = model.module
        self._params_lr: list[_OptParams] = []
        self.base_lr = learning_scheme.lr
        self.scheduler = learning_scheme.scheduler
        self.optimizer: torch.optim.Optimizer = learning_scheme.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_opt_params()),
            **learning_scheme.optimizer_defaults,
        )
        exp_dir = tracking.Experiment.current().dir
        self._checkpoint = checkpoint.CheckpointIO(model,
                                                   self.optimizer,
                                                   par_dir=exp_dir)

    def get_opt_params(self) -> list[_OptParams]:
        """
        Learning rates for each parameter updated according to the scheduler
        and the current epoch.
        """
        return [
            dict(params=g['params'], lr=self.get_scheduled_lr(g['lr']))
            for g in self._params_lr
        ]

    def get_scheduled_lr(self, lr: float) -> float:
        """
        Update the base learning rate according to the scheduler.

        Args:
            lr: base learning rate.
        """
        return self.scheduler(lr, self.model.epoch)

    @property
    def base_lr(self) -> float | dict[str, float]:
        """
        The learning rate(s) for the module parameters.
        """
        return self._lr

    @base_lr.setter
    def base_lr(self, lr: float | dict[str, float]) -> None:
        self._lr = lr
        if isinstance(lr, (float, int)):
            self._params_lr = [
                dict(params=self.module.parameters(), lr=lr),
            ]
        else:
            self._params_lr = [
                dict(params=getattr(self.module, k).parameters(), lr=v)
                for k, v in lr.items()
            ]
            if not self._params_lr_contains_all_params():
                raise exceptions.MissingParamError(repr(self.module), list(lr))
        return

    def update_learning_rate(
            self,
            lr: Optional[float | dict[str, float]] = None,
            scheduler: Optional[p.SchedulerProtocol] = None,
    ) -> None:
        """
        It updates the learning rates for each parameters' group in the
        optimizer based on input learning rate(s) and scheduler.

        Args:
            lr: learning rates for named parameters or global value. Default
                keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.
        """
        if scheduler is not None:
            self.scheduler = scheduler
        if lr is not None:
            self.base_lr = lr
        for g, up_g in zip(self.optimizer.param_groups,
                           self.get_opt_params()):
            g['lr'] = up_g['lr']
        return

    def load(self, epoch: int = -1) -> None:
        """Load model and optimizer state from a checkpoint."""
        self._checkpoint.load(epoch=epoch)

    def save(self) -> None:
        """Save model and optimizer state in a checkpoint."""
        self._checkpoint.save()

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(count_params(elem['params'])
                              for elem in self._params_lr)
        total_params_model = count_params(self.module.parameters())
        return total_params_lr == total_params_model

    def __repr__(self) -> str:
        desc = '{}(module={}, optimizer={})'
        return desc.format(self.__class__.__name__,
                           self.optimizer.__class__.__name__)


def count_params(params: Iterator) -> int:
    """Counts the number of parameters."""
    return sum(1 for _ in params)

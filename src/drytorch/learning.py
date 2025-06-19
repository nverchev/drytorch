"""Module containing classes for wrapping a torch module and its optimizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Literal, Optional, TypedDict, TypeVar, cast

import torch
from torch import cuda
import dataclasses

from drytorch import checkpointing
from drytorch import exceptions
from drytorch import protocols as p
from drytorch import registering
from drytorch import schedulers
from drytorch.utils import repr_utils

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
        base_lr: initial learning rates for named parameters or global value.
        optimizer_defaults: optional arguments for the optimizer.
        scheduler: modifies the learning rate given the current epoch.
        max_grad_norm: clips
    """
    optimizer_cls: type[torch.optim.Optimizer]
    base_lr: float | dict[str, float]
    scheduler: p.SchedulerProtocol = schedulers.ConstantScheduler()
    optimizer_defaults: dict[str, Any] = dataclasses.field(default_factory=dict)
    gradient_clipping: Literal['none', 'norm', 'value'] = 'none'
    gradient_threshold: float = 1.

    def __post_init__(self):
        if self.gradient_threshold <= 0:
            raise ValueError('Gradient treshold must be positive.')

    def clip_gradients_(self, parameters: Iterator[torch.nn.Parameter]):
        if self.gradient_clipping == 'norm':
            return torch.nn.utils.clip_grad_value_(parameters)
        elif self.gradient_clipping == 'value':
            return torch.nn.utils.clip_grad_norm_(parameters)
        return

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


class Model(repr_utils.Versioned, p.ModelProtocol[_Input_contra, _Output_co]):
    """
    Wrapper for a torch.nn.Module class with extra information.

    Attributes:
        module: Pytorch module to optimize.
        epoch: the number of epochs the model has been trained so far.
    """
    _name = repr_utils.DefaultName()

    def __init__(
            self,
            torch_module: p.ModuleProtocol[_Input_contra, _Output_co],
            /,
            name: str = '',
            device: Optional[torch.device] = None,
            checkpoint: p.CheckpointProtocol = checkpointing.LocalCheckpoint()
    ) -> None:
        """
        Args:
            torch_module: Pytorch module with type annotations.
            name: the name of the model. Default uses the class name.
            device: the device where to store the weights of module.
                Default uses cuda when available, cpu otherwise.

        """
        super().__init__()
        self.module = self._validate_module(torch_module)
        self._name = name
        self.epoch: int = 0
        self.device = self._default_device() if device is None else device
        registering.register_model(self)
        self.checkpoint = checkpoint
        self.checkpoint.register_model(self)

    def __call__(self, inputs: _Input_contra) -> _Output_co:
        return self.module(inputs)

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.module.to(device)
        return

    @property
    def name(self):
        """The name of the model."""
        return self._name

    def increment_epoch(self) -> None:
        """Increment the epoch by 1."""
        self.epoch += 1

    def load_state(self, epoch=-1) -> None:
        """Load the weights and epoch of the model"""
        self.checkpoint.load(epoch=epoch)

    def save_state(self) -> None:
        """Save the weights and epoch of the model"""
        self.checkpoint.save()

    def to(self, device: torch.device) -> None:
        """Forward the homonymous method."""
        self.device = device

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

    It supports different learning rates to separate parameters' groups.

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
            learning_scheme: p.LearningProtocol,
    ) -> None:
        self.model = model
        self.module = model.module
        self._params_lr: list[_OptParams] = []
        self.base_lr = learning_scheme.base_lr
        self.scheduler = learning_scheme.scheduler
        self.optimizer: torch.optim.Optimizer = learning_scheme.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_opt_params()),
            **learning_scheme.optimizer_defaults,
        )
        self.checkpoint = self.model.checkpoint
        self.checkpoint.register_optimizer(self.optimizer)

    def __repr__(self) -> str:
        desc = '{}(module={}, optimizer={})'
        return desc.format(self.__class__.__name__,
                           self.model.name,
                           self.optimizer.__class__.__name__)

    @property
    def base_lr(self) -> float | dict[str, float]:
        """
        Learning rate(s) for the module parameters.
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
                module_names = list(self.module.named_modules())
                raise exceptions.MissingParamError(module_names, list(lr))

        return

    def get_opt_params(self) -> list[_OptParams]:
        """
        Learning rates for each parameter updated according to the scheduler
        and the current epoch.
        """
        return [
            _OptParams(params=g['params'], lr=self.get_scheduled_lr(g['lr']))
            for g in self._params_lr
        ]

    def get_scheduled_lr(self, lr: float) -> float:
        """
        Update the base learning rate according to the scheduler.

        Args:
            lr: base learning rate.
        """
        return self.scheduler(lr, self.model.epoch)

    def load(self, epoch: int = -1) -> None:
        """Load model and optimizer state from a checkpoint."""
        self.checkpoint.load(epoch=epoch)

    def update_learning_rate(
            self,
            base_lr: Optional[float | dict[str, float]] = None,
            scheduler: Optional[p.SchedulerProtocol] = None,
    ) -> None:
        """
        Update the learning rates for each parameter's group in the
        optimizer based on input learning rate(s) and scheduler.

        Args:
            base_lr: initial learning rates for named parameters or global
                value. Default keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.
        """
        if scheduler is not None:
            self.scheduler = scheduler

        if base_lr is not None:
            self.base_lr = base_lr

        for g, up_g in zip(self.optimizer.param_groups,
                           self.get_opt_params()):
            g['lr'] = up_g['lr']

        return

    def save(self) -> None:
        """Save model and optimizer state in a checkpoint."""
        self.checkpoint.save()

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(count_params(elem['params'])
                              for elem in self._params_lr)
        total_params_model = count_params(self.module.parameters())
        return total_params_lr == total_params_model


def count_params(params: Iterator) -> int:
    """Count the number of parameters."""
    return sum(1 for _ in params)

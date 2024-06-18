from __future__ import annotations

from functools import wraps
from typing import Optional, Callable, Concatenate, Self, Iterable, Iterator
from typing import ParamSpec, Final, TypeVar, Any, cast
import copy

import pandas as pd
import torch
from torch import cuda

from dry_torch import checkpoint
from dry_torch import exceptions
from dry_torch import schedulers
from dry_torch import protocols
from dry_torch import data_types
from dry_torch import tracking

_Input_contra = TypeVar('_Input_contra',
                        bound=data_types.InputType,
                        contravariant=True)
_Target_contra = TypeVar('_Target_contra',
                         bound=data_types.InputType,
                         contravariant=True)
_Input = TypeVar('_Input', bound=data_types.InputType)
_Target = TypeVar('_Target', bound=data_types.TargetType)
_Output = TypeVar('_Output', bound=data_types.OutputType)

_P = ParamSpec('_P')
_RT = TypeVar('_RT')


class Network(protocols.NetworkProtocol[_Input_contra, _Output]):
    """
    Bundle the module and its optimizer.
    Support different learning rates and separate parameters groups.

    Args:
        torch_module: Pytorch module to optimize. AnnotatedModule type variables
        match the class type variables.
         Defaults to torch.optim.Adam.
        for a global value.
        (same for all the parameters).
        value does not implement a scheduler.
        device: the device where to store the module. Default uses cuda when
        available else cpu.


    Attributes:
        module: Pytorch module to optimize. Annotate forward method for better
         linting
        device: the device where to store the module.

    Methods:
        clone(): return a deepcopy of the instance.
        params_lr: property for the updated learning rates for the optimizer.
        update_learning_rate(): update the optimizer with the updated settings
         or with an input learning rate.
    """

    def __init__(
            self,
            torch_module: protocols.ModuleProtocol[_Input_contra, _Output],
            /,
            name: str = 'network',
            device: Optional[torch.device] = None,
    ) -> None:
        self.name: Final = name
        self.device = self.default_device() if device is None else device
        self.module = self.validate_module(torch_module).to(self.device)
        self.checkpoint_io = checkpoint.CheckpointIO(self)
        self.epoch = 0
        self.log: data_types.LogsDict = {
            split: pd.DataFrame() for split in data_types.Split
        }

        exp = tracking.Experiment.current()
        exp.register_module(torch_module, name)

    @staticmethod
    def validate_module(torch_model) -> torch.nn.Module:
        if not isinstance(torch_model, torch.nn.Module):
            raise TypeError('torch_module must be a torch.nn.Module subclass')
        return torch_model

    @staticmethod
    def default_device():
        return torch.device('cuda:0' if cuda.is_available() else 'cpu')

    def compile(
            self,
            loss_calc: protocols.LossCallable[_Output, _Target_contra],
            optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
            lr: float | dict[str, float] = 0.001,
            other_optimizer_args: Optional[dict[str, Any]] = None,
            scheduler: protocols.SchedulerProtocol = (
                    schedulers.ConstantScheduler()
            ),
    ) -> Model[_Input_contra, _Target_contra, _Output]:
        return Model(
            self,
            loss_calc=loss_calc,
            optimizer_cls=optimizer_cls,
            lr=lr,
            other_optimizer_args=other_optimizer_args,
            scheduler=scheduler,
        )

    def _copy_module(  # type: ignore
            self
    ) -> protocols.ModuleProtocol[_Input_contra, _Output]:
        return copy.deepcopy(self.module)

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

    def save(self):
        self.checkpoint_io.save()

    def load(self, epoch=-1):
        self.checkpoint_io.load(epoch=epoch)

    def __call__(self, inputs: _Input_contra) -> _Output:
        return self.module(inputs)


class Model(protocols.ModelProtocol[_Input, _Target, _Output]):
    """
    Bundle the module and its optimizer.
    Support different learning rates and separate parameters groups.

    Args:
        match the class type variables.
        optimizer_cls: the optimizer class to bind to the module.
         Defaults to torch.optim.Adam.
        lr: a dictionary of learning rates for the named parameters or a float
        for a global value.
        other_optimizer_args: optional arguments for the optimizer
        (same for all the parameters).
        scheduler: modifies the learning rate given the current epoch. Default
        value does not implement a scheduler.


    Attributes:

        optimizer: the optimizer bound to the module.

    Methods:
        clone(): return a deepcopy of the instance.
        params_lr: property for the updated learning rates for the optimizer.
        update_learning_rate(): update the optimizer with the updated settings
         or with an input learning rate.
    """

    def __init__(
            self,
            network: protocols.NetworkProtocol[_Input, _Output],
            /,
            *,
            loss_calc: protocols.LossCallable[_Output, _Target],
            optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
            lr: float | dict[str, float] = 0.001,
            other_optimizer_args: Optional[dict[str, Any]] = None,
            scheduler: protocols.SchedulerProtocol = (
                    schedulers.ConstantScheduler()
            ),
    ) -> None:
        self.network = network
        self.loss_calc = loss_calc
        self._params_lr: list[protocols.OptParams] = []
        self.set_lr(lr)
        self.scheduler = scheduler
        self.optimizer = optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_scheduled_lr()),
            **({} if other_optimizer_args is None else other_optimizer_args),
        )
        self.checkpoint_io = checkpoint.CheckpointIO(network, self.optimizer)

    def get_base_lr(self) -> list[protocols.OptParams]:
        return self._params_lr

    def get_scheduled_lr(self) -> list[protocols.OptParams]:
        """
        Learning rates for each parameter updated according to the scheduler
        and the current epoch.
        """
        exp = tracking.Experiment.current()
        epoch = exp.model[self.network.name].epoch
        return [
            dict(params=g['params'], lr=self.scheduler(g['lr'], epoch))
            for g in self._params_lr
        ]

    def to(self, device: torch.device):
        self.network.device = device
        self.network.module.to(device)

    # params_lr.setter does not implement typing suggestion
    def set_lr(self, lr: float | dict[str, float]) -> None:
        """
        Pair the learning rates with the parameter groups.

        Args:
            lr: a dictionary of learning rates for the named parameters or
            a float for a global value.
        """
        if isinstance(lr, (float, int)):
            self._params_lr = [
                dict(params=self.network.module.parameters(), lr=lr),
            ]
        else:
            self._params_lr = [
                dict(params=getattr(self.network.module, k).parameters(), lr=v)
                for k, v in lr.items()
            ]
            if not self._params_lr_contains_all_params():
                raise exceptions.MissingParamError(
                    repr(self.network.module), list(lr)
                )
        return

    def update_learning_rate(
            self,
            lr: Optional[float | dict[str, float]] = None
    ) -> None:
        """
        It updates the learning rates for each parameters' group in the
        optimizer.

        Args:
            lr: a dictionary of learning rates for the named parameters or a
            float for a global value.
               If None (default), the scheduled original learning rates are
               used. Else, the scheduler is deactivated.
        """
        if lr is not None:
            self.set_lr(lr)
            self.scheduler = schedulers.ConstantScheduler()
        for g, up_g in zip(self.optimizer.param_groups,
                           self.get_scheduled_lr()):
            g['lr'] = up_g['lr']
        return

    def clone(self, new_name: str) -> Self:
        """
        Return a copy of the deepcopy of the object.
        """
        cloned = self.__class__(
            self.network.clone(new_name),
            loss_calc=self.loss_calc,
            optimizer_cls=type(self.optimizer),
            other_optimizer_args=self.optimizer.defaults,
            scheduler=self.scheduler,
        )
        cloned._params_lr = self.get_base_lr()
        cloned.update_learning_rate()
        return cloned

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(self.count_params(elem['params'])
                              for elem in self._params_lr)
        total_params_model = self.count_params(self.network.module.parameters())
        return total_params_lr == total_params_model

    def __repr__(self):
        desc = '{}(module={}, optimizer={})'
        return desc.format(self.__class__.__name__,
                           self.network.module.name,
                           self.optimizer.__class__.__name__)

    @staticmethod
    def count_params(params: Iterator) -> int:
        return sum(1 for _ in params)

    def save(self):
        self.checkpoint_io.save()

    def load(self, epoch=-1):
        self.checkpoint_io.load(epoch=epoch)


def bind_to_model(
        func: Callable[
            Concatenate[
                Any,
                protocols.ModelProtocol[_Input, _Target, _Output],
                _P
            ],
            _RT
        ]
) -> Callable[
    Concatenate[
        Any,
        protocols.ModelProtocol[_Input, _Target, _Output],
        _P],
    _RT
]:
    """
    Decorator that extracts metadata from a function named arguments.

    Args:
        func: the function that we want to extract metadata from.
    Returns:
        Callable: the same input function.
    """

    @wraps(func)
    def wrapper(instance: Any,
                compiled: protocols.ModelProtocol,
                *args: _P.args,
                **kwargs: _P.kwargs) -> _RT:
        network = compiled.network
        if not isinstance(network, protocols.NetworkProtocol):
            raise exceptions.BoundedModelTypeError(network)
        exp = tracking.Experiment.current()
        cls = instance.__class__
        if cls in exp.model[network.name].bindings:
            raise exceptions.AlreadyBoundedError(network.name, cls)
        exp.model[network.name].bindings[cls] = instance
        tracking.add_metadata(exp, network.name, kwargs)
        return func(instance, compiled, *args, **kwargs)

    return wrapper

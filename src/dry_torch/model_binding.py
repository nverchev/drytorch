from __future__ import annotations

from functools import wraps
from typing import Optional, Callable, Concatenate, Self, Iterable, Iterator
from typing import ParamSpec, Final, TypeVar, Any, cast
import copy
import torch
from torch import cuda
from dry_torch import exceptions
from dry_torch import schedulers
from dry_torch import protocols
from dry_torch import data_types
from dry_torch import tracking

_Input_contra = TypeVar('_Input_contra',
                        bound=data_types.InputType,
                        contravariant=True)
_Output_co = TypeVar('_Output_co',
                     bound=data_types.OutputType,
                     covariant=True)
_P = ParamSpec('_P')
_RT = TypeVar('_RT')


class ModelOptimizer(
    protocols.ModelOptimizerProtocol[_Input_contra, _Output_co]
):
    """
    Bundle the model and its optimizer.
    Support different learning rates and separate parameters groups.

    Args:
        torch_model: Pytorch model to optimize. AnnotatedModule type variables
        match the class type variables.
        optimizer_cls: the optimizer class to bind to the model.
         Defaults to torch.optim.Adam.
        lr: a dictionary of learning rates for the named parameters or a float
        for a global value.
        other_optimizer_args: optional arguments for the optimizer
        (same for all the parameters).
        scheduler: modifies the learning rate given the current epoch. Default
        value does not implement a scheduler.
        device: the device where to store the model. Default uses cuda when
        available else cpu.


    Attributes:
        model: Pytorch module to optimize. Annotate forward method for better
         linting
        optimizer: the optimizer bound to the model.
        device: the device where to store the model.

    Methods:
        clone(): return a deepcopy of the instance.
        params_lr: property for the updated learning rates for the optimizer.
        update_learning_rate(): update the optimizer with the updated settings
         or with an input learning rate.
    """

    def __new__(
            cls,
            torch_model: protocols.ModuleProtocol[_Input_contra, _Output_co],
            name: str = 'model',
            **kwargs: Any,
    ) -> ModelOptimizer[_Input_contra, _Output_co]:
        exp = tracking.Experiment.get_active_environment()
        exp.register_model(torch_model, name)
        tracking.add_metadata(exp, name, kwargs)
        return cast(ModelOptimizer, super().__new__(cls))

    def __init__(
            self,
            torch_model: protocols.ModuleProtocol[_Input_contra, _Output_co],
            /,
            *,
            name='model',
            optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
            lr: float | dict[str, float] = 0.001,
            other_optimizer_args: Optional[dict[str, Any]] = None,
            scheduler: protocols.SchedulerProtocol = (
                    schedulers.ConstantScheduler()
            ),
            device: Optional[torch.device] = None,
    ) -> None:

        self.name: Final = name
        if device is None:
            device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
        self.device = device
        if not isinstance(torch_model, torch.nn.Module):
            raise TypeError('torch_model must be a torch.nn.Module subclass')
        self.model = torch_model.to(self.device)
        self._params_lr: list[protocols.OptParams] = []
        self.set_lr(lr)
        self.scheduler = scheduler
        self.optimizer = optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_scheduled_lr()),
            **({} if other_optimizer_args is None else other_optimizer_args),
        )

    def get_base_lr(self) -> list[protocols.OptParams]:
        return self._params_lr

    def get_scheduled_lr(self) -> list[protocols.OptParams]:
        """
        Learning rates for each parameter updated according to the scheduler
        and the current epoch.
        """
        exp = tracking.Experiment.get_active_environment()
        epoch = exp.model[self.name].epoch
        return [
            dict(params=g['params'], lr=self.scheduler(g['lr'], epoch))
            for g in self._params_lr
        ]

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)

    # params_lr.setter does not implement typing suggestion
    def set_lr(self, lr: float | dict[str, float]) -> None:
        """
        Pair the learning rates with the parameter groups.

        Args:
            lr: a dictionary of learning rates for the named parameters or
            a float for a global value.
        """
        if isinstance(lr, float):
            self._params_lr = [
                dict(params=self.model.parameters(), lr=lr),
            ]
        else:
            self._params_lr = [
                dict(params=getattr(self.model, k).parameters(), lr=v)
                for k, v in lr.items()
            ]
            if not self._params_lr_contains_all_params():
                raise exceptions.MissingParamError(repr(self.model), list(lr))
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

    def __call__(self,
                 inputs: _Input_contra) -> _Output_co:
        return self.model.forward(inputs)

    def clone(self, new_name: str) -> Self:
        """
        Return a copy of the deepcopy of the object.
        """
        cloned_model: protocols.ModuleProtocol[
            _Input_contra,
            _Output_co
        ]
        cloned_model = copy.deepcopy(self.model)
        cloned = self.__class__(
            cloned_model,
            name=new_name,
            lr=0.001,
            optimizer_cls=type(self.optimizer),
            other_optimizer_args=self.optimizer.defaults,
            scheduler=self.scheduler,
            device=self.device,
        )
        cloned._params_lr = self.get_base_lr()
        cloned.update_learning_rate()
        return cloned

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(self.count_params(elem['params'])
                              for elem in self._params_lr)
        total_params_model = self.count_params(self.model.parameters())
        return total_params_lr == total_params_model

    def __repr__(self):
        desc = '{}(model={}, optimizer={})'
        return desc.format(self.__class__.__name__,
                           self.model,
                           self.optimizer.__class__.__name__)

    @staticmethod
    def count_params(params: Iterator) -> int:
        return sum(1 for _ in params)


def bind_to_model(
        func: Callable[
            Concatenate[
                Any,
                protocols.ModelOptimizerProtocol[_Input_contra, _Output_co],
                _P
            ],
            _RT
        ]
) -> Callable[
    Concatenate[
        Any,
        protocols.ModelOptimizerProtocol[_Input_contra, _Output_co],
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
                model_optim: protocols.ModelOptimizerProtocol,
                *args: _P.args,
                **kwargs: _P.kwargs) -> _RT:
        if not isinstance(model_optim, protocols.ModelOptimizerProtocol):
            raise exceptions.BoundedModelTypeError(model_optim)
        exp = tracking.Experiment.get_active_environment()
        cls = instance.__class__
        if cls in exp.model[model_optim.name].bindings:
            raise exceptions.AlreadyBoundedError(model_optim.name, cls)
        exp.model[model_optim.name].bindings[cls] = instance
        tracking.add_metadata(exp, model_optim.name, kwargs)
        return func(instance, model_optim, *args, **kwargs)

    return wrapper

from __future__ import annotations

from functools import wraps
from typing import Optional, Callable, Concatenate, Self, Iterable, Iterator
from typing import ParamSpec, TypeVar, Any, cast, Generic, Type
import copy
import torch
from torch import cuda
import dataclasses

from dry_torch import io
from dry_torch import exceptions
from dry_torch import scheduling
from dry_torch import protocols as p
from dry_torch import tracking

_Input_contra = TypeVar('_Input_contra',
                        bound=p.InputType,
                        contravariant=True)
_Target_contra = TypeVar('_Target_contra',
                         bound=p.InputType,
                         contravariant=True)
_Output_co = TypeVar('_Output_co',
                     bound=p.OutputType,
                     covariant=True)

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)

_P = ParamSpec('_P')
_RT = TypeVar('_RT')


@dataclasses.dataclass
class LearningScheme(p.LearningProtocol):
    """
        optimizer_cls: the optimizer class to bind_to_model to the module.
         Defaults to torch.optim.Adam.
        lr: a dictionary of learning rates for the named parameters or a float
        for a global value.
        optimizer_defaults: optional arguments for the optimizer
        (same for all the parameters).
        scheduler: modifies the learning rate given the current epoch. Default
        value does not implement a scheduler.
    """

    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam
    lr: float | dict[str, float] = 0.001
    scheduler: p.SchedulerProtocol = scheduling.ConstantScheduler()
    optimizer_defaults: dict[str, Any] = dataclasses.field(default_factory=dict)


class Model(Generic[_Input_contra, _Output_co]):
    default_model_name = tracking.DefaultName('Model')
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
            torch_module: p.ModuleProtocol[_Input_contra, _Output_co],
            /,
            name: Optional[str] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        self.name: str = name or self.__class__.default_model_name()
        self.device = self.default_device() if device is None else device
        self.module = self.validate_module(torch_module).to(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.checkpoint = io.ModelStateIO(self)

    @staticmethod
    def validate_module(torch_model) -> torch.nn.Module:
        if not isinstance(torch_model, torch.nn.Module):
            raise TypeError('torch_module must be a torch.nn.Module subclass')
        return torch_model

    @staticmethod
    def default_device() -> torch.device:
        return torch.device('cuda:0' if cuda.is_available() else 'cpu')

    def _copy_module(
            self
    ) -> p.ModuleProtocol[_Input_contra, _Output_co]:
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

    def save_state(self) -> None:
        self.checkpoint.save()

    def load_state(self, epoch=-1) -> None:
        self.checkpoint.load(epoch=epoch)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.module.to(device)

    def __call__(self, inputs: _Input_contra) -> _Output_co:
        return self.module(inputs)


class ModelOptimizer:
    """
    Bundle the module and its optimizer.
    Support different learning rates and separate parameters groups.

    Args:

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
            model: p.ModelProtocol[_Input_contra, _Output_co],
            learning_scheme: p.LearningProtocol = LearningScheme()
    ) -> None:
        self.model = model
        self.module = self.model.module
        self.learning_scheme = learning_scheme
        self._params_lr: list[p.OptParams] = []
        self.set_lr(learning_scheme.lr)
        self.scheduler = learning_scheme.scheduler
        self.optimizer: torch.optim.Optimizer = learning_scheme.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_scheduled_lr()),
            **learning_scheme.optimizer_defaults,
        )

    def get_scheduled_lr(self) -> list[p.OptParams]:
        """
        Learning rates for each parameter updated according to the scheduler
        and the current epoch.
        """
        epoch = self.get_epoch()
        return [
            dict(params=g['params'], lr=self.scheduler(g['lr'], epoch))
            for g in self._params_lr
        ]

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
                dict(params=self.module.parameters(), lr=lr),
            ]
        else:
            self._params_lr = [
                dict(params=getattr(self.module, k).parameters(), lr=v)
                for k, v in lr.items()
            ]
            if not self._params_lr_contains_all_params():
                raise exceptions.MissingParamError(
                    repr(self.module), list(lr)
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
            self.scheduler = scheduling.ConstantScheduler()
        for g, up_g in zip(self.optimizer.param_groups,
                           self.get_scheduled_lr()):
            g['lr'] = up_g['lr']
        return

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(self.count_params(elem['params'])
                              for elem in self._params_lr)
        total_params_model = self.count_params(self.module.parameters())
        return total_params_lr == total_params_model

    def __repr__(self) -> str:
        desc = '{}(module={}, optimizer={})'
        return desc.format(self.__class__.__name__,
                           self.optimizer.__class__.__name__)

    @staticmethod
    def count_params(params: Iterator) -> int:
        return sum(1 for _ in params)

    def get_epoch(self) -> int:
        return tracking.Experiment.current().tracking[self.model.name].epoch


def bind_to_model(
        func: Callable[
            Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P],
            _RT],
) -> Callable[
    Concatenate[Any, p.ModelProtocol[_Input_contra, _Output_co], _P],
    _RT,
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
                model: p.ModelProtocol[_Input_contra, _Output_co],
                *args: _P.args,
                **kwargs: _P.kwargs) -> _RT:
        if not isinstance(model, p.ModelProtocol):
            raise exceptions.BoundedModelTypeError(model)
        exp = tracking.Experiment.current()
        model_tracking = exp.tracking[model.name]
        bindings = model_tracking.bindings
        cls_str = instance.__class__.__name__
        if cls_str in bindings:
            raise exceptions.AlreadyBoundedError(model.name, cls_str)
        cls_count = bindings.setdefault(cls_str, tracking.DefaultName(cls_str))
        tracking.add_metadata(exp, model.name, cls_count(), kwargs)
        return func(instance, model, *args, **kwargs)

    return wrapper


def unbind(instance: Any,
           model: p.ModelProtocol[_Input_contra, _Output_co]) -> None:
    if not isinstance(model, p.ModelProtocol):
        raise exceptions.BoundedModelTypeError(model)
    model_tracking = tracking.Experiment.current().tracking[model.name]
    metadata = model_tracking.metadata
    cls_str = instance.__class__.__name__
    if cls_str not in model_tracking.bindings:
        raise exceptions.NotBoundedError(model.name, cls_str)
    cls_str_counter = repr(model_tracking.bindings.pop(cls_str))
    metadata[cls_str_counter]['model_final_epoch'] = model_tracking.epoch
    return

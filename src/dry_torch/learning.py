from collections.abc import Iterable, Iterator
from typing import Optional, Self, TypeVar, Any, cast, Generic
import copy
import torch
from torch import cuda
import dataclasses

from src.dry_torch import descriptors
from src.dry_torch import exceptions
from src.dry_torch import schedulers
from src.dry_torch import protocols as p
from src.dry_torch import tracking
from src.dry_torch import registering

_Input_contra = TypeVar('_Input_contra',
                        bound=p.InputType,
                        contravariant=True)
_Output_co = TypeVar('_Output_co',
                     bound=p.OutputType,
                     covariant=True)


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



class Model(Generic[_Input_contra, _Output_co]):
    _default_model_name = tracking.DefaultName('Model', start=0)
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
        self.module = self.validate_module(torch_module)
        self.name: str = name or Model._default_model_name()
        self.epoch: int = 0
        self.device = device
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.checkpoint = io.ModelStateIO(self)
        registering.register_model(self)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: Optional[torch.device]) -> None:
        device = self.default_device() if device is None else device
        self._device = device
        self.module.to(device)
        return

    def increment_epoch(self) -> None:
        self.epoch += 1

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
            learning_scheme: p.LearningProtocol
    ) -> None:
        self.model = model
        self.module = self.model.module
        self.learning_scheme = learning_scheme
        self._params_lr: list[descriptors.OptParams] = []
        self.set_lr(learning_scheme.lr)
        self.scheduler = learning_scheme.scheduler
        self.optimizer: torch.optim.Optimizer = learning_scheme.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_scheduled_lr()),
            **learning_scheme.optimizer_defaults,
        )

    def get_scheduled_lr(self) -> list[descriptors.OptParams]:
        """
        Learning rates for each parameter updated according to the scheduler
        and the current epoch.
        """
        epoch = self.model.epoch
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
            self.scheduler = schedulers.ConstantScheduler()
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


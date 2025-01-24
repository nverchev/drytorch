"""Classes for wrapping a torch module and its optimizer."""

from collections.abc import Iterable, Iterator
from typing import Optional, Self, TypeVar, TypedDict, Any, cast
import copy
import torch
from torch import cuda
import dataclasses

from src.dry_torch import repr_utils
from src.dry_torch import exceptions
from src.dry_torch import schedulers
from src.dry_torch import protocols as p
from src.dry_torch import checkpoint

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
    _default_model_name = repr_utils.DefaultName()

    def __init__(
            self,
            torch_module: p.ModuleProtocol[_Input_contra, _Output_co],
            /,
            name: Optional[str] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        self.module = self._validate_module(torch_module)
        self.name: str = name or self._default_model_name
        self.epoch: int = 0
        self.device = self._default_device() if device is None else device
        self._model_state_io = checkpoint.ModelStateIO(self)

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
        self.set_lr(learning_scheme.lr)
        self.scheduler = learning_scheme.scheduler
        self.optimizer: torch.optim.Optimizer = learning_scheme.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_scheduled_lr()),
            **learning_scheme.optimizer_defaults,
        )
        self._checkpoint = checkpoint.CheckpointIO(model, self.optimizer)

    def get_scheduled_lr(self) -> list[_OptParams]:
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
                raise exceptions.MissingParamError(repr(self.module), list(lr))
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
                float for a global value. If None (default), the scheduled
                original learning rates are used.
                Else, the scheduler is deactivated.
        """
        if lr is not None:
            self.set_lr(lr)
            self.scheduler = schedulers.ConstantScheduler()
        for g, up_g in zip(self.optimizer.param_groups,
                           self.get_scheduled_lr()):
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

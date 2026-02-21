"""Module containing classes for wrapping a torch module and its optimizer."""

from __future__ import annotations

import abc
import sys

from collections.abc import Callable
from typing import ClassVar, Final, TypeVar

import torch

from torch import distributed as dist
from typing_extensions import override

from drytorch.core import protocols as p
from drytorch.core import registering
from drytorch.lib import checkpoints
from drytorch.utils import repr_utils


__all__ = [
    'AveragedModel',
    'EMAModel',
    'Model',
    'SWAModel',
]

Input = TypeVar('Input', bound=p.InputType, contravariant=True)
Output = TypeVar('Output', bound=p.OutputType, covariant=True)
Tensor = torch.Tensor
_ParamList = tuple[Tensor, ...] | list[Tensor]
_MultiAvgFn = Callable[[_ParamList, _ParamList, Tensor | int], None]


class Model(repr_utils.CreatedAtMixin, p.ModelProtocol[Input, Output]):
    """Wrapper for a torch.nn.Module class with extra information.

    Attributes:
        exec_module: Pytorch module used for execution.
        epoch: the number of epochs the model has been trained so far.
        mixed_precision: whether to use mixed precision computing.
        checkpoint: checkpoint manager.
    """

    _name = repr_utils.DefaultName()

    exec_module: torch.nn.Module
    epoch: int
    mixed_precision: bool
    checkpoint: p.CheckpointProtocol
    _device: torch.device
    _should_compile: bool
    _should_dist: bool
    _registered: bool

    def __init__(  # type: ignore
        self,
        module: p.ModuleProtocol[Input, Output],
        name: str = '',
        device: torch.device | None = None,
        checkpoint: p.CheckpointProtocol | None = None,
        mixed_precision: bool = False,
        should_compile: bool = True,
        should_distribute: bool = True,
    ) -> None:
        """Initialize.

        Option should_distribute assumes that there is a single accelerator for
        each process and that the device for the process is already set.

        Args:
            module: Pytorch module with type annotations.
            name: the name of the model. Default uses the class name.
            device: the device where to store the weights of the module.
                Default uses the accelerator if available, cpu otherwise.
            checkpoint: class that saves the state and optionally the optimizer.
            mixed_precision: whether to use mixed precision computing.
            should_compile: compile the module at instantiation (Python < 3.14).
            should_distribute: wrap the module for data-distributed settings.
        """
        super().__init__()
        self._device = self._default_device() if device is None else device
        self._should_compile = should_compile
        self._should_dist = should_distribute
        self.mixed_precision: Final = mixed_precision
        torch_module = self._validate_module(module)
        self.exec_module: Final = self.prepare_module(torch_module)
        self._name = name
        self.epoch = 0
        if checkpoint is None:
            checkpoint = checkpoints.LocalCheckpoint()

        self.checkpoint = checkpoint
        self.checkpoint.bind_model(self)
        self._registered = False
        self.register()
        return

    def __call__(self, inputs: Input) -> Output:
        """Execute forward pass."""
        with torch.autocast(
            device_type=self.device.type, enabled=self.mixed_precision
        ):
            return self.exec_module(inputs)

    def __del__(self):
        """Unregister from the registry when deleted/garbage-collected."""
        try:
            self.unregister()
        except AttributeError:  # may happen during instantiation
            pass

        return

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""
        return self._device

    @property
    def module(self) -> torch.nn.Module:
        """The module wrapped by the class."""
        return self._unwrap_module()

    @property
    def name(self) -> str:
        """The name of the model."""
        return self._name

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Compile and distribute the module."""
        module = module.to(self._device)

        # TODO: remove flag when torch.compile is supported on Python 3.14
        if self._should_compile and sys.version_info < (3, 14):
            torch.compile(module)

        if dist.is_available() and dist.is_initialized() and self._should_dist:
            if self._device.type == 'cuda':
                module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)

            if self.device.index is not None:
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[self.device.index]
                )
            else:
                module = torch.nn.parallel.DistributedDataParallel(module)

        return module

    def increment_epoch(self) -> None:
        """Increment the epoch by 1."""
        self.epoch += 1

    def load_state(self, epoch=-1) -> None:
        """Load the weights and epoch of the model."""
        self.checkpoint.load(epoch=epoch)

    def register(self) -> None:
        """Register to the registry."""
        registering.register_model(self)
        self._registered = True
        return

    def save_state(self) -> None:
        """Save the weights and epoch of the model."""
        self.checkpoint.save()

    def unregister(self) -> None:
        """Unregister from the registry."""
        if self._registered:
            registering.unregister_model(self)

        self._registered = False
        return

    def _unwrap_module(self) -> torch.nn.Module:
        """Return the module without wrapping."""
        wrapper_types = (
            torch.nn.DataParallel,
            torch.nn.parallel.DistributedDataParallel,
        )
        if isinstance(self.exec_module, wrapper_types):
            return self.exec_module.module

        return self.exec_module

    def post_batch_update(self) -> None:
        """Update the model after processing a batch of data."""
        return

    def post_epoch_update(self) -> None:
        """Update the model after processing an epoch of data."""
        return

    @staticmethod
    def _default_device() -> torch.device:
        device = torch.accelerator.current_accelerator()
        if device is not None:
            index = torch.accelerator.current_device_index()
            return torch.device(device.type, index)

        return torch.device('cpu')

    @staticmethod
    def _validate_module(
        torch_model: p.ModuleProtocol[Input, Output],
    ) -> torch.nn.Module:
        if not isinstance(torch_model, torch.nn.Module):
            raise TypeError('torch_module must be a torch.nn.Module subclass')

        return torch_model


class AveragedModel(Model[Input, Output], abc.ABC):
    """Bundle a torch.nn.Module and a torch.optim.swa_utils.AveragedModel.

    Use the averaged model when in inference mode.

    Attributes:
        averaged_module: the averaged module.
    """

    average_name: ClassVar[str] = 'averaged_model'
    averaged_module: torch.optim.swa_utils.AveragedModel

    def __init__(
        self,
        torch_module: p.ModuleProtocol[Input, Output],
        /,
        name: str = '',
        device: torch.device | None = None,
        checkpoint: p.CheckpointProtocol | None = None,
        mixed_precision: bool = False,
    ) -> None:
        """Initialize.

        Args:
            torch_module: Pytorch module with type annotations.
            name: the name of the model. Default uses the class name.
            device: the device where to store the weights of the module.
                Default uses cuda when available, cpu otherwise.
            checkpoint: class that saves the state and optionally the optimizer.
            mixed_precision: whether to use mixed precision computing.
                Defaults to False.
        """
        super().__init__(
            torch_module, name, device, checkpoint, mixed_precision
        )
        self.averaged_module = self._create_averaged_module()
        self.checkpoint.bind_module(self.average_name, self.averaged_module)
        return

    def __call__(self, inputs: Input) -> Output:
        """Execute the forward pass."""
        if torch.is_inference_mode_enabled():
            return self.averaged_module(inputs)  # no mixed precision here

        return super().__call__(inputs)

    def _create_averaged_module(self) -> torch.optim.swa_utils.AveragedModel:
        averaged_module = torch.optim.swa_utils.AveragedModel(
            self.module, self.device, multi_avg_fn=self._get_multi_avg_fn()
        )
        averaged_module.eval()
        for param in averaged_module.parameters():
            param.requires_grad_(False)

        return averaged_module

    @abc.abstractmethod
    def _get_multi_avg_fn(self) -> _MultiAvgFn | None: ...

    def _update_parameters(self) -> None:
        self.averaged_module.update_parameters(self._unwrap_module())
        return


class SWAModel(AveragedModel[Input, Output]):
    """Bundle a torch.nn.Module and a torch.optim.swa_utils.AveragedModel.

    Use the averaged model when in inference mode.

    Attributes:
        averaged_module: the averaged module.
        start_epoch: the epoch at which to start averaging.
    """

    average_name = 'swa_model'
    averaged_module: torch.optim.swa_utils.AveragedModel

    def __init__(
        self,
        torch_module: p.ModuleProtocol[Input, Output],
        /,
        start_epoch: int,
        name: str = '',
        device: torch.device | None = None,
        checkpoint: p.CheckpointProtocol | None = None,
        mixed_precision: bool = False,
    ) -> None:
        """Initialize.

        Args:
            torch_module: Pytorch module with type annotations.
            start_epoch: the epoch at which to start averaging.
            name: the name of the model. Default uses the class name.
            device: the device where to store the weights of the module.
                Default uses cuda when available, cpu otherwise.
            checkpoint: class that saves the state and optionally the optimizer.
            mixed_precision: whether to use mixed precision computing.
                Defaults to False.
        """
        self.start_epoch: Final = start_epoch
        super().__init__(
            torch_module, name, device, checkpoint, mixed_precision
        )
        return

    def __call__(self, inputs: Input) -> Output:
        """Execute the forward pass."""
        if torch.is_inference_mode_enabled() and self.epoch >= self.start_epoch:
            return self.averaged_module(inputs)  # no mixed precision here

        return super(AveragedModel, self).__call__(inputs)

    @override
    def post_epoch_update(self) -> None:
        if self.epoch >= self.start_epoch:
            self._update_parameters()

        return

    @override
    def _get_multi_avg_fn(self) -> None:
        return None


class EMAModel(AveragedModel[Input, Output]):
    """Bundle a torch.nn.Module and a torch.optim.swa_utils.AveragedModel.

    Use the averaged model when in inference mode.

    Attributes:
        averaged_module: the averaged module.
        decay: the exponential decay rate for the moving average.
    """

    average_name = 'ema_model'
    averaged_module: torch.optim.swa_utils.AveragedModel
    decay: float

    def __init__(
        self,
        torch_module: p.ModuleProtocol[Input, Output],
        /,
        name: str = '',
        device: torch.device | None = None,
        checkpoint: p.CheckpointProtocol | None = None,
        mixed_precision: bool = False,
        decay: float = 0.999,
    ) -> None:
        """Initialize.

        Args:
            torch_module: Pytorch module with type annotations.
            name: the name of the model. Default uses the class name.
            device: the device where to store the weights of the module.
                Default uses cuda when available, cpu otherwise.
            checkpoint: class that saves the state and optionally the optimizer.
            mixed_precision: whether to use mixed precision computing.
                Defaults to False.
            decay: the exponential decay rate for the moving average.
        """
        self.decay: Final = decay
        super().__init__(
            torch_module, name, device, checkpoint, mixed_precision
        )
        return

    @override
    def _get_multi_avg_fn(self) -> _MultiAvgFn:
        return torch.optim.swa_utils.get_ema_multi_avg_fn(decay=self.decay)

    @override
    def post_batch_update(self) -> None:
        self._update_parameters()
        return

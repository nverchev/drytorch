"""Classes for wrapping a torch module and its optimizer."""

from __future__ import annotations

import abc
import typing

from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
)

import torch

from torch import distributed as dist
from torch.nn import parallel
from typing_extensions import override

from drytorch.core import log_events, registering
from drytorch.core import protocols as p
from drytorch.lib import checkpoints
from drytorch.utils import repr_utils


__all__ = [
    'AveragedModel',
    'EMAModel',
    'Model',
    'ModuleProtocol',
    'SWAModel',
]

Input = TypeVar('Input', bound=p.InputType, contravariant=True)
Output = TypeVar('Output', bound=p.OutputType, covariant=True)
Tensor = torch.Tensor


_ParamList = tuple[Tensor, ...] | list[Tensor]
_MultiAvgFn = Callable[[_ParamList, _ParamList, Tensor | int], None]


class ModuleProtocol(Protocol[Input, Output]):
    """Protocol for a PyTorch module with type annotations."""

    @abc.abstractmethod
    def forward(self, inputs: Input, /) -> Output:
        """Forward run of the network."""


class Model(repr_utils.CreatedAtMixin, Generic[Input, Output]):
    """Wrapper for a torch.nn.Module class with extra information.

    Attributes:
        checkpoint: saves and loads the state of the model.
        exec_module: :class:`torch.nn.Module` used for execution.
        mixed_precision: the autocast precision; ``None`` disables autocast.
    """

    _name = repr_utils.DefaultName()

    checkpoint: Final[p.CheckpointProtocol]
    exec_module: Final[torch.nn.Module]
    mixed_precision: Final[torch.dtype | None]
    _compile: bool
    _device: torch.device
    _epoch: int
    _distribute: bool
    _registered: bool

    def __init__(
        self,
        module: ModuleProtocol[Input, Output],
        /,
        *,
        checkpoint: p.CheckpointProtocol | None = None,
        device: torch.device | Literal['auto'] = 'auto',
        distribute: bool = True,
        mixed_precision: torch.dtype | None = None,
        name: str = '',
        torch_compile: bool = False,
    ) -> None:
        """Initialize.

        The distribute option assumes that there is a single accelerator for
        each process and that the device for the process is already set.

        Args:
            module: :class:`torch.nn.Module` with type annotations.
            checkpoint: manages saving and loading of model weights.
                Defaults to :class:`~drytorch.lib.checkpoints.LocalCheckpoint`.
            device: the device where to store the weights of the module.
                Default uses the available accelerator, or the CPU otherwise.
            distribute: whether to wrap the module for ddp training.
            mixed_precision: the autocast precision. Default disables
                autocast.
            name: the name of the model. Default uses the class name.
            torch_compile: whether to compile the module with torch.compile.

        Raises:
            ValueError: if ``mixed_precision`` is not None, bfloat16 or float16.
        """
        super().__init__()
        self._device = _get_auto_device() if device == 'auto' else device
        self._compile = torch_compile
        self._distribute = distribute
        if mixed_precision not in (None, torch.float16, torch.bfloat16):
            msg = 'mixed_precision must be torch.float16 or torch.bfloat16.'
            raise ValueError(msg)

        self.mixed_precision = mixed_precision
        validated_module = self._validate_module(module)
        self.exec_module = self.prepare_module(validated_module)
        self._name = name
        self._epoch = 0
        self._registered = False
        self.register()
        if checkpoint is None:
            checkpoint = checkpoints.LocalCheckpoint()

        self.checkpoint = checkpoint
        self.checkpoint.bind_model(self)
        return

    def __call__(self, inputs: Input, /) -> Output:
        """Execute forward pass."""
        if torch.is_inference_mode_enabled():
            module = self._get_inference_module()
        else:
            module = self.exec_module

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision,
            enabled=self.mixed_precision is not None,
        ):
            return module(inputs)

    @property
    def device(self) -> torch.device:
        """The device where the weights are stored."""
        return self._device

    @property
    def epoch(self) -> int:
        """The number of epochs the model has been trained so far."""
        return self._epoch

    @property
    def module(self) -> torch.nn.Module:
        """The module wrapped by the class."""
        return self._unwrap_module()

    @property
    def name(self) -> str:
        """The name of the model."""
        return self._name

    def __del__(self):
        """Unregister from the registry when deleted/garbage-collected."""
        try:
            self.unregister()
        except AttributeError:  # may happen during instantiation
            pass

        return

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Compile and distribute the module."""
        module = module.to(self._device)
        is_compiled = isinstance(
            module, torch._dynamo.eval_frame.OptimizedModule
        )
        if self._compile and not is_compiled:
            module = typing.cast(torch.nn.Module, torch.compile(module))

        if dist.is_available() and dist.is_initialized() and self._distribute:
            if self._device.type == 'cuda':
                module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)

            if self.device.index is not None:
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[self.device.index]
                )
            else:
                module = torch.nn.parallel.DistributedDataParallel(module)

        return module

    def bind_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Bind the optimizer whose state is saved with the model.

        Args:
            optimizer: the optimizer updating the model parameters.

        Raises:
            OptimizerAlreadyBoundError: if another optimizer is bound.
        """
        self.checkpoint.bind_optimizer(optimizer)
        return

    def increment_epoch(self) -> None:
        """Increment the epoch by 1.

        Side Effects:
            Emits :class:`~drytorch.core.log_events.IncrementEpochEvent`.
        """
        self._epoch += 1
        log_events.IncrementEpochEvent(model_name=self.name, epoch=self.epoch)
        return

    def load_state(self, epoch: int = -1) -> None:
        """Load the weights, the optimizer state and the epoch of the model.

        Args:
            epoch: the epoch to load. Default loads the last saved epoch.
        """
        self._epoch = self.checkpoint.load(epoch=epoch)
        return

    def register(self) -> None:
        """Register to the registry."""
        registering.register_model(self)
        self._registered = True
        return

    def save_state(self) -> None:
        """Save the weights and epoch of the model."""
        self.checkpoint.save()
        return

    def unregister(self) -> None:
        """Unregister from the registry."""
        if self._registered:
            registering.unregister_model(self)

        self._registered = False
        return

    def post_batch_update(self) -> None:
        """Update the model after processing a batch of data."""
        return

    def post_epoch_update(self) -> None:
        """Update the model after processing an epoch of data."""
        return

    def _get_inference_module(self) -> torch.nn.Module:
        """Return the module used in inference mode."""
        return self._unwrap_ddp()

    def _unwrap_ddp(self) -> torch.nn.Module:
        """Return the module without the ddp wrapper."""
        module = self.exec_module
        if isinstance(module, parallel.DistributedDataParallel):
            module = cast(torch.nn.Module, module.module)

        return module

    def _unwrap_module(self) -> torch.nn.Module:
        """Return the module without wrapping."""
        module = self._unwrap_ddp()
        if isinstance(module, torch._dynamo.eval_frame.OptimizedModule):
            module = module._orig_mod

        return module

    @staticmethod
    def _validate_module(
        module: ModuleProtocol[Any, Any],
    ) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Module):
            raise TypeError('module must be a torch.nn.Module subclass')

        return module


class AveragedModel(Model[Input, Output], abc.ABC):
    """Bundle a torch.nn.Module and a torch.optim.swa_utils.AveragedModel.

    Use the averaged model when in inference mode.

    Attributes:
        average_name: name identifier for the averaged model.
        exec_averaged_module: the averaged module.
    """

    average_name: ClassVar[str] = 'averaged_model'
    exec_averaged_module: Final[torch.optim.swa_utils.AveragedModel]
    _inference_averaged_module: torch.nn.Module

    def __init__(
        self,
        module: ModuleProtocol[Input, Output],
        /,
        *,
        checkpoint: p.CheckpointProtocol | None = None,
        device: torch.device | Literal['auto'] = 'auto',
        distribute: bool = True,
        mixed_precision: torch.dtype | None = None,
        name: str = '',
        torch_compile: bool = False,
    ) -> None:
        """Initialize.

        Args:
            module: :class:`torch.nn.Module` with type annotations.
            checkpoint: manages saving and loading of model weights.
                Defaults to :class:`~drytorch.lib.checkpoints.LocalCheckpoint`.
            device: the device where to store the weights of the module.
                Default uses the available accelerator, or the CPU otherwise.
            distribute: whether to wrap the module for ddp training.
            mixed_precision: the autocast precision. Default disables
                autocast.
            name: the name of the model. Default uses the class name.
            torch_compile: whether to compile the module with torch.compile.

        Raises:
            ValueError: if ``mixed_precision`` is not None, bfloat16 or float16.
        """
        super().__init__(
            module,
            checkpoint=checkpoint,
            device=device,
            distribute=distribute,
            mixed_precision=mixed_precision,
            name=name,
            torch_compile=torch_compile,
        )
        self.exec_averaged_module = self._create_averaged_module()
        self.checkpoint.bind_module(
            self.average_name,
            self.exec_averaged_module,  # save wrapped module
        )
        self._inference_averaged_module = self.averaged_module
        if self._compile:  # compiled copy shares the averaged parameters
            self._inference_averaged_module = cast(
                torch.nn.Module, torch.compile(self.averaged_module)
            )

        return

    @property
    def averaged_module(self) -> torch.nn.Module:
        """The module wrapped by the class."""
        return self._unwrap_averaged_module()

    def _create_averaged_module(self) -> torch.optim.swa_utils.AveragedModel:
        averaged_module = torch.optim.swa_utils.AveragedModel(
            self.module,
            self.device,
            multi_avg_fn=self._get_multi_avg_fn(),
            use_buffers=True,
        )
        averaged_module.eval()
        for param in averaged_module.parameters():
            param.requires_grad_(False)

        return averaged_module

    @abc.abstractmethod
    def _get_multi_avg_fn(self) -> _MultiAvgFn | None:
        """Define the averaging function for the model parameters."""

    @override
    def _get_inference_module(self) -> torch.nn.Module:
        return self._inference_averaged_module

    def _update_parameters(self) -> None:
        self.exec_averaged_module.update_parameters(self._unwrap_module())
        return

    def _unwrap_averaged_module(self) -> torch.nn.Module:
        return self.exec_averaged_module.module


class SWAModel(AveragedModel[Input, Output]):
    """Bundle a torch.nn.Module and a torch.optim.swa_utils.AveragedModel.

    Use the averaged model when in inference mode.

    Attributes:
        exec_averaged_module: the averaged module.
        start_epoch: the epoch at which to start averaging.
    """

    average_name: ClassVar[str] = 'swa_model'
    start_epoch: Final[int]

    def __init__(
        self,
        module: ModuleProtocol[Input, Output],
        /,
        *,
        checkpoint: p.CheckpointProtocol | None = None,
        device: torch.device | Literal['auto'] = 'auto',
        distribute: bool = True,
        mixed_precision: torch.dtype | None = None,
        name: str = '',
        torch_compile: bool = False,
        start_epoch: int,
    ) -> None:
        """Initialize.

        Args:
            module: :class:`torch.nn.Module` with type annotations.
            checkpoint: manages saving and loading of model weights.
                Defaults to :class:`~drytorch.lib.checkpoints.LocalCheckpoint`.
            device: the device where to store the weights of the module.
                Default uses the available accelerator, or the CPU otherwise.
            distribute: whether to wrap the module for ddp training.
            mixed_precision: the autocast precision. Default disables
                autocast.
            name: the name of the model. Default uses the class name.
            torch_compile: whether to compile the module with torch.compile.
            start_epoch: the epoch at which to start averaging.

        Raises:
            ValueError: if ``mixed_precision`` is not None, bfloat16 or float16.
        """
        self.start_epoch = start_epoch
        super().__init__(
            module,
            checkpoint=checkpoint,
            device=device,
            distribute=distribute,
            mixed_precision=mixed_precision,
            name=name,
            torch_compile=torch_compile,
        )
        return

    @override
    def post_epoch_update(self) -> None:
        if self.epoch >= self.start_epoch:
            self._update_parameters()

        return

    @override
    def _get_inference_module(self) -> torch.nn.Module:
        if self.epoch < self.start_epoch:
            return self._unwrap_ddp()

        return super()._get_inference_module()

    @override
    def _get_multi_avg_fn(self) -> None:
        return None


class EMAModel(AveragedModel[Input, Output]):
    """Bundle a torch.nn.Module and a torch.optim.swa_utils.AveragedModel.

    Use the averaged model when in inference mode.

    Attributes:
        decay: the exponential decay rate for the moving average.
        exec_averaged_module: the averaged module.
    """

    average_name: ClassVar[str] = 'ema_model'
    decay: Final[float]

    def __init__(
        self,
        module: ModuleProtocol[Input, Output],
        /,
        *,
        checkpoint: p.CheckpointProtocol | None = None,
        device: torch.device | Literal['auto'] = 'auto',
        distribute: bool = True,
        mixed_precision: torch.dtype | None = None,
        name: str = '',
        torch_compile: bool = False,
        decay: float = 0.999,
    ) -> None:
        """Initialize.

        Args:
            module: :class:`torch.nn.Module` with type annotations.
            checkpoint: manages saving and loading of model weights.
                Defaults to :class:`~drytorch.lib.checkpoints.LocalCheckpoint`.
            device: the device where to store the weights of the module.
                Default uses the available accelerator, or the CPU otherwise.
            distribute: whether to wrap the module for ddp training.
            mixed_precision: the autocast precision. Default disables
                autocast.
            name: the name of the model. Default uses the class name.
            torch_compile: whether to compile the module with torch.compile.
            decay: the exponential decay rate for the moving average.

        Raises:
            ValueError: if ``mixed_precision`` is not None, bfloat16 or float16.
        """
        self.decay = decay
        super().__init__(
            module,
            checkpoint=checkpoint,
            device=device,
            distribute=distribute,
            mixed_precision=mixed_precision,
            name=name,
            torch_compile=torch_compile,
        )
        return

    @override
    def post_batch_update(self) -> None:
        self._update_parameters()
        return

    @override
    def _get_multi_avg_fn(self) -> _MultiAvgFn:
        return torch.optim.swa_utils.get_ema_multi_avg_fn(decay=self.decay)


def _get_auto_device() -> torch.device:
    device = torch.accelerator.current_accelerator()
    if device is not None and torch.accelerator.is_available():
        index = torch.accelerator.current_device_index()
        return torch.device(device.type, index)

    return torch.device('cpu')

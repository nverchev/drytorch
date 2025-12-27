"""Module containing classes for wrapping a torch module and its optimizer."""

from __future__ import annotations

import sys

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Final, TypedDict, TypeVar, cast

import torch

from torch import amp
from torch import distributed as dist
from typing_extensions import override

from drytorch.core import exceptions, register
from drytorch.core import protocols as p
from drytorch.lib import checkpoints
from drytorch.utils import repr_utils


__all__ = [
    'Model',
    'ModelAverage',
    'ModelOptimizer',
]


Input = TypeVar('Input', bound=p.InputType, contravariant=True)
Output = TypeVar('Output', bound=p.OutputType, covariant=True)
Tensor = torch.Tensor
ParamList = tuple[Tensor, ...] | list[Tensor]


class _OptParams(TypedDict):
    params: Iterator[torch.nn.Parameter]
    lr: float


class Model(repr_utils.CreatedAtMixin, p.ModelProtocol[Input, Output]):
    """Wrapper for a torch.nn.Module class with extra information.

    Attributes:
        module: Pytorch module to optimize.
        epoch: the number of epochs the model has been trained so far.
        mixed_precision: whether to use mixed precision computing.
    """

    _name = repr_utils.DefaultName()

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
        """Constructor.

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
        self.mixed_precision = mixed_precision
        torch_module = self._validate_module(module)
        self.module = self.prepare_module(torch_module)
        self._name = name
        self.epoch: int = 0
        if checkpoint is None:
            checkpoint = checkpoints.LocalCheckpoint()

        self.checkpoint: p.CheckpointProtocol = checkpoint
        self.checkpoint.bind_model(self)
        self._registered: bool = False
        self.register()
        return

    def __call__(self, inputs: Input) -> Output:
        """Execute forward pass."""
        with torch.autocast(
            device_type=self.device.type, enabled=self.mixed_precision
        ):
            return self.module(inputs)

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
    def name(self) -> str:
        """The name of the model."""
        return self._name

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Compile and distribute the module."""
        module = module.to(self._default_device())

        # TODO: remove flag when torch.compile is supported on Python 3.14
        if self._should_compile and sys.version_info < (3, 14):
            torch.compile(module)

        if dist.is_available and dist.is_initialized() and self._should_dist:
            batch_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
            module = torch.nn.parallel.DistributedDataParallel(batch_module)

        return module

    def increment_epoch(self) -> None:
        """Increment the epoch by 1."""
        self.epoch += 1

    def load_state(self, epoch=-1) -> None:
        """Load the weights and epoch of the model."""
        self.checkpoint.load(epoch=epoch)

    def register(self) -> None:
        """Register to the registry."""
        register.register_model(self)
        self._registered = True
        return

    def save_state(self) -> None:
        """Save the weights and epoch of the model."""
        self.checkpoint.save()

    def unregister(self) -> None:
        """Unregister from the registry."""
        if self._registered:
            register.unregister_model(self)

        self._registered = False
        return

    def update_parameters(self) -> None:
        """Update the parameters of the model."""
        return

    @staticmethod
    def _default_device() -> torch.device:
        device_or_none = torch.accelerator.current_accelerator()
        return torch.device('cpu') if device_or_none is None else device_or_none

    @staticmethod
    def _validate_module(
        torch_model: p.ModuleProtocol[Input, Output],
    ) -> torch.nn.Module:
        if not isinstance(torch_model, torch.nn.Module):
            raise TypeError('torch_module must be a torch.nn.Module subclass')

        return torch_model


class ModelAverage(Model[Input, Output]):
    """Bundle a torch.nn.Module and a torch.optim.swa_utils.AveragedModel.

    Use the averaged model when in inference mode.

    Attributes:
        module: Pytorch module to optimize.
        epoch: the number of epochs the model has been trained so far.
    """

    _default_checkpoint = checkpoints.LocalCheckpoint()

    def __init__(
        self,
        torch_module: p.ModuleProtocol[Input, Output],
        /,
        name: str = '',
        device: torch.device | None = None,
        checkpoint: p.CheckpointProtocol = _default_checkpoint,
        mixed_precision: bool = False,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = None,
        multi_avg_fn: Callable[[ParamList, ParamList, Tensor | int], None]
        | None = None,
        use_buffers: bool = False,
    ) -> None:
        """Constructor.

        Args:
            torch_module: Pytorch module with type annotations.
            name: the name of the model. Default uses the class name.
            device: the device where to store the weights of the module.
                Default uses cuda when available, cpu otherwise.
            checkpoint: class that saves the state and optionally the optimizer.
            mixed_precision: whether to use mixed precision computing.
                Defaults to False.
            avg_fn: see docs at torch.optim.swa_utils.AveragedModel.
            multi_avg_fn: see docs at torch.optim.swa_utils.AveragedModel.
            use_buffers: see docs at torch.optim.swa_utils.AveragedModel.
        """
        super().__init__(
            torch_module, name, device, checkpoint, mixed_precision
        )
        self.averaged_module = torch.optim.swa_utils.AveragedModel(
            self.module, self.device, avg_fn, multi_avg_fn, use_buffers
        )
        return

    def __call__(self, inputs: Input) -> Output:
        """Execute the forward pass."""
        if torch.inference_mode():
            return self.averaged_module(inputs)  # no mixed precision here
        return super().__call__(inputs)

    @override
    def update_parameters(self) -> None:
        """Update the parameters of the model."""
        self.averaged_module.update_parameters(self.module)
        return


class ModelOptimizer:
    """Bundle the module and its optimizer.

    It supports different learning rates to separate parameters' groups.
    """

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        learning_schema: p.LearningProtocol,
    ) -> None:
        """Constructor.

        Args:
            model: the model to be optimized.
            learning_schema: the learning scheme for the optimizer.
        """
        self._model: Final = model
        self._module: Final = model.module
        self._lr: float | dict[str, float] = {}
        self._params_lr: list[_OptParams] = []
        self.base_lr = learning_schema.base_lr
        self._scheduler = learning_schema.scheduler
        self._optimizer: torch.optim.Optimizer = learning_schema.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_opt_params()),
            **learning_schema.optimizer_defaults,
        )
        self._gradient_op: p.GradientOpProtocol = learning_schema.gradient_op
        self._checkpoint: p.CheckpointProtocol = self._model.checkpoint
        self._checkpoint.bind_optimizer(self._optimizer)
        self._scaler: amp.grad_scaler.GradScaler = amp.grad_scaler.GradScaler(
            model.device.type,
            enabled=model.mixed_precision,
        )

    @override
    def __repr__(self) -> str:
        desc = '{}(module={}, optimizer={})'
        return desc.format(
            self.__class__.__name__,
            self._model.name,
            self._optimizer.__class__.__name__,
        )

    @property
    def base_lr(self) -> float | dict[str, float]:
        """Learning rate(s) for the module parameters."""
        return self._lr

    @base_lr.setter
    def base_lr(self, lr: float | dict[str, float]) -> None:
        self._lr = lr
        if isinstance(lr, float | int):
            self._params_lr = [
                {'params': self._module.parameters(), 'lr': lr},
            ]
        else:
            self._params_lr = [
                {'params': getattr(self._module, k).parameters(), 'lr': v}
                for k, v in lr.items()
            ]
            if not self._params_lr_contains_all_params():
                module_names: list[str] = [
                    named_elem[0] for named_elem in self._module.named_modules()
                ]
                raise exceptions.MissingParamError(module_names, list(lr))

        return

    def get_opt_params(self) -> list[_OptParams]:
        """Actual learning rates for each parameter updated according."""
        return [
            _OptParams(params=g['params'], lr=self.get_scheduled_lr(g['lr']))
            for g in self._params_lr
        ]

    def get_scheduled_lr(self, lr: float) -> float:
        """Update the base learning rate according to the scheduler.

        Args:
            lr: base learning rate.
        """
        return self._scheduler(lr, self._model.epoch)

    def load(self, epoch: int = -1) -> None:
        """Load model and optimizer state from a checkpoint."""
        self._checkpoint.load(epoch=epoch)

    def update_learning_rate(
        self,
        base_lr: float | dict[str, float] | None = None,
        scheduler: p.SchedulerProtocol | None = None,
    ) -> None:
        """Recalculate the learning rates for the current epoch.

        It updates the learning rates for each parameter's group in the
        optimizer based on input learning rate(s) and scheduler.

        Args:
            base_lr: initial learning rates for named parameters or global
                value. Default keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.
        """
        if scheduler is not None:
            self._scheduler = scheduler

        if base_lr is not None:
            self.base_lr = base_lr

        for g, up_g in zip(
            self._optimizer.param_groups, self.get_opt_params(), strict=False
        ):
            g['lr'] = up_g['lr']

        return

    def optimize(self, loss_value: Tensor):
        """Optimize the model backpropagating the loss value.

        Args:
            loss_value: the output tensor for the loss.
        """
        self._scaler.scale(loss_value).backward()
        self._gradient_op(self._model.module.parameters())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad()

    def save(self) -> None:
        """Save model and optimizer state in a checkpoint."""
        self._checkpoint.save()

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(
            count_params(elem['params']) for elem in self._params_lr
        )
        total_params_model = count_params(self._module.parameters())
        return total_params_lr == total_params_model


def count_params(params: Iterator[Any]) -> int:
    """Count the number of parameters."""
    return sum(1 for _ in params)

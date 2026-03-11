"""Module containing classes for training a model."""

from __future__ import annotations

import warnings

from collections.abc import Iterable, Iterator
from typing import Any, Final, Self, TypedDict, TypeVar, cast

import torch

from torch import amp
from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.core import protocols as p
from drytorch.lib import evaluations, hooks, runners


__all__ = [
    'ModelOptimizer',
    'Trainer',
]


class _OptParams(TypedDict):
    params: Iterator[torch.nn.Parameter]
    lr: float


Input = TypeVar('Input', bound=p.InputType)
Target = TypeVar('Target', bound=p.TargetType)
Output = TypeVar('Output', bound=p.OutputType)


class ModelOptimizer:
    """Bundle the module and its optimizer.

    It supports different learning rates to separate parameters' groups.
    """

    _model: p.ModelProtocol
    _module: torch.nn.Module
    _lr: float | dict[str, float]
    _params_lr: list[_OptParams]
    _scheduler: p.SchedulerProtocol
    _optimizer: torch.optim.Optimizer
    _gradient_op: p.GradientOpProtocol
    _checkpoint: p.CheckpointProtocol
    _scaler: amp.grad_scaler.GradScaler

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        learning_schema: p.LearningProtocol,
    ) -> None:
        """Initialize.

        Args:
            model: the model to be optimized.
            learning_schema: the learning scheme for the optimizer.
        """
        self._model: Final = model
        self._module: Final = model.module
        self._lr = {}
        self._params_lr = []
        self.base_lr = learning_schema.base_lr
        self._scheduler = learning_schema.scheduler
        self._optimizer = learning_schema.optimizer_cls(
            params=cast(Iterable[dict[str, Any]], self.get_opt_params()),
            **learning_schema.optimizer_defaults,
        )
        self._gradient_op = learning_schema.gradient_op
        self._model.checkpoint.bind_optimizer(self._optimizer)
        self._scaler = amp.grad_scaler.GradScaler(
            model.device.type,
            enabled=model.mixed_precision,
        )
        return

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
        """Learning rate(s) for the module parameters.

        Raises:
            MissingParamError: if parameters are missing from the dictionary.
        """
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

    def optimize(self, loss_value: torch.Tensor):
        """Optimize the model backpropagating the loss value.

        Args:
            loss_value: the output tensor for the loss.
        """
        self._optimizer.zero_grad()
        self._scaler.scale(loss_value).backward()
        self._scaler.unscale_(self._optimizer)
        self._gradient_op(self._model.module.parameters())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        return

    def _params_lr_contains_all_params(self) -> bool:
        total_params_lr = sum(
            self._count_params(elem['params']) for elem in self._params_lr
        )
        total_params_model = self._count_params(self._module.parameters())
        return total_params_lr == total_params_model

    @staticmethod
    def _count_params(params: Iterator[Any]) -> int:
        """Count the number of parameters."""
        return sum(1 for _ in params)


class Trainer(
    runners.ModelRunnerWithLogs[
        Input, Target, Output, p.LossProtocol[Output, Target]
    ],
    p.TrainerProtocol[Input, Target, Output],
):
    """Implement the standard Pytorch training loop.

    Attributes:
        model: the model to train.
        loader: provides inputs and targets in batches.
        objective: determines the optimization's criterion.
        learning_schema: contains optimizer settings and scheduling.
        validation: class that validates the model,
    """

    def __init__(
        self,
        model: p.ModelProtocol[Input, Output],
        name: str = '',
        *,
        loader: p.LoaderProtocol[tuple[Input, Target]],
        loss: p.LossProtocol[Output, Target],
        learning_schema: p.LearningProtocol,
    ) -> None:
        """Initialize.

        Args:
            model: the model containing the weights to evaluate.
            name: the base name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            loader: provides inputs and targets in batches.
            loss: determines the optimization's criterion.
            learning_schema: contains optimizer settings and scheduling.
        """
        super().__init__(model, loader=loader, objective=loss, name=name)
        self.learning_schema: Final = learning_schema
        self.validation: p.MonitorProtocol | None = None
        self._model_optimizer: Final = ModelOptimizer(model, learning_schema)
        self.pre_epoch_hooks: Final = hooks.HookRegistry[
            Trainer[Input, Target, Output]
        ]()
        self.post_epoch_hooks: Final = hooks.HookRegistry[
            Trainer[Input, Target, Output]
        ]()
        self._terminated = False
        return

    @property
    @override
    def terminated(self) -> bool:
        return self._terminated

    @override
    def __call__(self, store_outputs: bool = False) -> None:
        """Train the module for one epoch.

        Args:
            store_outputs: whether to store model outputs.
        """
        if self.terminated:
            warnings.warn(exceptions.TerminatedTrainingWarning(), stacklevel=1)
            return

        self.model.module.train()
        self.model.increment_epoch()
        self._model_optimizer.update_learning_rate()
        try:
            super().__call__()
        except exceptions.ConvergenceError as ce:
            self.terminate_training(reason=str(ce))
            raise ce

        self.model.post_epoch_update()
        return

    def add_validation(
        self,
        val_loader: p.LoaderProtocol[tuple[Input, Target]],
        name: str = '',
        interval: int = 1,
    ) -> None:
        """Add a loader for validation with the same metrics as for training.

        If different validation loaders are added, they will all be performed,
        but only the last will be stored as the instance validation.

        Args:
            val_loader: the loader for validation.
            name: the name for the validation.
            interval: the frequency of validation.

        Raises:
            ValueError: if the interval is not strictly positive.
        """
        validation = evaluations.Validation(
            self.model, name=name, loader=val_loader, metric=self.objective
        )
        val_hook = hooks.StaticHook(validation)
        if interval < 1:
            raise ValueError(f'Interval must larger than 0. Got {interval}.')

        if interval > 1:
            val_hook.bind(hooks.call_every(interval))

        self.post_epoch_hooks.register(val_hook)
        self.validation = validation
        return

    @override
    def load_checkpoint(self, epoch: int = -1) -> None:
        """Load model and optimizer state from a checkpoint.

        Args:
            epoch: the epoch from which to load the checkpoint.
                Defaults to the last saved epoch.
        """
        self.model.checkpoint.load(epoch=epoch)
        return

    @override
    def save_checkpoint(self) -> None:
        self.model.checkpoint.save()

    @override
    def terminate_training(self, reason: str) -> None:
        self._terminated = True
        log_events.TerminatedTrainingEvent(
            source_name=self.name,
            model_name=self.model.name,
            epoch=self.model.epoch,
            reason=reason,
        )
        return

    @override
    def train(self, n_epochs: int) -> None:
        if self.terminated:
            warnings.warn(exceptions.TerminatedTrainingWarning(), stacklevel=1)
            return
        final_epoch = self.model.epoch + n_epochs
        log_events.StartTrainingEvent(
            source_name=self.name,
            model_name=self.model.name,
            start_epoch=self.model.epoch,
            end_epoch=final_epoch,
        )
        for _ in range(n_epochs):
            self.pre_epoch_hooks.execute(self)
            log_events.StartEpochEvent(
                source_name=self.name,
                model_name=self.model.name,
                epoch=self.model.epoch + 1,
                end_epoch=final_epoch,
            )
            self()
            self.post_epoch_hooks.execute(self)
            log_events.EndEpochEvent(
                source_name=self.name,
                model_name=self.model.name,
                epoch=self.model.epoch,
            )
            if self.terminated:
                break

        log_events.EndTrainingEvent(self.name)
        return

    def train_until(self: Self, epoch: int) -> None:
        """Train the module until the specified epoch.

        Args:
            epoch: the final epoch in the training.

        """
        remaining_epochs = epoch - self.model.epoch
        if remaining_epochs > 0:
            self.train(remaining_epochs)

        if remaining_epochs < 0:
            warnings.warn(
                exceptions.PastEpochWarning(epoch, self.model.epoch),
                stacklevel=1,
            )
        return

    @override
    def update_learning_rate(
        self,
        base_lr: float | dict[str, float] | None = None,
        scheduler: p.SchedulerProtocol | None = None,
    ) -> None:
        """Update the learning rate(s).

        It updates the learning rates for each parameter's group in the
        optimizer based on input learning rate(s) and scheduler.

        Args:
            base_lr: initial learning rates for named parameters or global
                value. Default keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.
        """
        scheduler_name = None if scheduler is None else repr(scheduler)
        log_events.LearningRateEvent(
            model_name=self.model.name,
            source_name=self.name,
            epoch=self.model.epoch,
            base_lr=base_lr,
            scheduler_name=scheduler_name,
        )
        self._model_optimizer.update_learning_rate(base_lr, scheduler)
        return

    @override
    def _run_backward(self, outputs: Output, targets: Target) -> None:
        # replace super call
        loss_value = self.objective.forward(outputs, targets)
        try:
            if torch.isinf(loss_value) or torch.isnan(loss_value):
                raise exceptions.ConvergenceError(loss_value.item())

        except RuntimeError as re:
            if loss_value.numel() != 1:
                raise exceptions.LossNotScalarError(loss_value.shape) from re

            raise re

        self._model_optimizer.optimize(loss_value)
        self.model.post_batch_update()
        return

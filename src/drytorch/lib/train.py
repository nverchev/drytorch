"""Module containing classes for training a model."""

import warnings

from typing import Final, Self, TypeVar

import torch

from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.core import protocols as p
from drytorch.lib import evaluations, hooks, models, runners


_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


class Trainer(
    runners.ModelRunnerWithLogs[
        _Input, _Target, _Output, p.LossProtocol[_Output, _Target]
    ],
    p.TrainerProtocol[_Input, _Target, _Output],
):
    """Implement the standard Pytorch training loop.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        learning_scheme: contains optimizer settings and scheduling.
        validation: class that validates the model,
    """

    def __init__(
        self,
        model: p.ModelProtocol[_Input, _Output],
        name: str = '',
        *,
        loader: p.LoaderProtocol[tuple[_Input, _Target]],
        loss: p.LossProtocol[_Output, _Target],
        learning_schema: p.LearningProtocol,
    ) -> None:
        """Constructor.

        Args:
            model: the model containing the weights to evaluate.
            name: the base name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            loader: provides inputs and targets in batches.
            loss: processes the model outputs and targets.
            learning_schema: contains optimizer settings and scheduling.
        """
        super().__init__(model, loader=loader, objective=loss, name=name)
        self.learning_schema = learning_schema
        self.validation: p.MonitorProtocol | None = None
        self._model_optimizer: Final = models.ModelOptimizer(
            model, learning_schema
        )
        self.pre_epoch_hooks: Final = hooks.HookRegistry[
            Trainer[_Input, _Target, _Output]
        ]()
        self.post_epoch_hooks: Final = hooks.HookRegistry[
            Trainer[_Input, _Target, _Output]
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

        return

    def add_validation(
        self, val_loader: p.LoaderProtocol[tuple[_Input, _Target]]
    ) -> None:
        """Add a loader for validation with the same metrics as for training.

        If different validation loaders are added, they will all be performed,
        but only the last will be stored as the instance validation.

        Args:
            val_loader: the loader for validation.
        """
        validation = evaluations.Validation(
            self.model, loader=val_loader, metric=self.objective
        )
        val_hook = hooks.StaticHook(validation)
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
        self._model_optimizer.load(epoch=epoch)

    @override
    def save_checkpoint(self) -> None:
        self._model_optimizer.save()

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
    def train(self, num_epochs: int) -> None:
        if self.terminated:
            warnings.warn(exceptions.TerminatedTrainingWarning(), stacklevel=1)
            return
        final_epoch = self.model.epoch + num_epochs
        log_events.StartTrainingEvent(
            source_name=self.name,
            model_name=self.model.name,
            start_epoch=self.model.epoch,
            end_epoch=final_epoch,
        )
        for _ in range(num_epochs):
            log_events.StartEpochEvent(
                source_name=self.name,
                model_name=self.model.name,
                epoch=self.model.epoch + 1,
                end_epoch=final_epoch,
            )
            self.pre_epoch_hooks.execute(self)
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

    @override
    def _run_backward(self, outputs: _Output, targets: _Target) -> None:
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
        self.model.update_parameters()

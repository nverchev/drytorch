"""Classes for training a model."""

import copy
from typing import Self, Optional, TypeVar
from typing_extensions import override
import warnings

import torch
from torch.cuda import amp

from dry_torch import evaluating
from dry_torch import exceptions
from dry_torch import learning
from dry_torch import log_events
from dry_torch import hooks
from dry_torch import protocols as p

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


class Trainer(evaluating.Evaluation[_Input, _Target, _Output],
              p.TrainerProtocol[_Input, _Target, _Output]):
    """
    Implement the standard Pytorch training loop.

    Attributes:
        model: the model containing the weights to evaluate.
        loader: provides inputs and targets in batches.
        calculator: processes the model outputs and targets.
        learning_scheme: contains optimizer settings and scheduling.
        mixed_precision: whether it uses mixed precision computing.
        outputs_list: list of optionally stored outputs

    """

    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            calculator: p.LossCalculatorProtocol[_Output, _Target],
            learning_scheme: p.LearningProtocol,
            name: str = '',
            mixed_precision: bool = False,
    ) -> None:
        """
        Args:
            model: the model containing the weights to evaluate.
            loader: provides inputs and targets in batches.
            calculator: processes the model outputs and targets.
            learning_scheme: contains optimizer settings and scheduling.
            name: the base name for the object for logging purposes.
                Defaults to class name plus eventual counter.
            mixed_precision: if allowed, use mixed precision computing.
        """
        super().__init__(model,
                         loader=loader,
                         calculator=calculator,
                         mixed_precision=mixed_precision,
                         name=name)
        self.model: p.ModelProtocol[_Input, _Output]
        self.calculator: p.LossCalculatorProtocol[_Output, _Target] = calculator
        self.learning_scheme = learning_scheme
        self.validation: Optional[evaluating.Validation] = None
        self._model_optimizer = learning.ModelOptimizer(model, learning_scheme)
        self._optimizer = self._model_optimizer.optimizer
        self._scaler = amp.GradScaler(enabled=mixed_precision)
        self.pre_epoch_hooks = hooks.HookRegistry[Self]()
        self.post_epoch_hooks = hooks.HookRegistry[Self]()
        self._terminated = False
        return

    @property
    def terminated(self) -> bool:
        """If true, this trainer should not be used for training anymore."""
        return self._terminated

    @override
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Train the module for one epoch.

        Args:
            store_outputs: whether to store model outputs.
        """
        super().__call__()
        if self.terminated:
            warnings.warn(exceptions.TerminatedTrainingWarning())
            return
        self.model.increment_epoch()
        self._model_optimizer.update_learning_rate()
        self.model.module.train()
        try:
            self._run_epoch(store_outputs)
        except exceptions.ConvergenceError as ce:
            self.terminate_training(reason=str(ce))
        return

    def add_validation(
            self,
            val_loader: p.LoaderProtocol[tuple[_Input, _Target]]
    ) -> None:
        """
        Add loader for validation with same metrics as for training.

        If different validation loaders are added, they will all be performed
        but only the last will be stored as the instance validation.

        Args:
            val_loader: the loader for validation.
        """
        calculator_copy = copy.deepcopy(self.calculator)
        validation = evaluating.Validation(self.model,
                                           loader=val_loader,
                                           calculator=calculator_copy)

        val_hook = hooks.StaticHook(validation)
        self.post_epoch_hooks.register(val_hook)
        self.validation = validation
        return

    def load_checkpoint(self, epoch: int = -1) -> None:
        """
        Load model and optimizer state from a checkpoint.

        Args:
            epoch: the epoch from which to load the checkpoint.
                Defaults to the last saved epoch.
        """
        self._model_optimizer.load(epoch=epoch)

    def save_checkpoint(self) -> None:
        """Save model and optimizer state in a checkpoint."""
        self._model_optimizer.save()

    def terminate_training(self, reason: str) -> None:
        """Prevent the trainer from continue the training."""
        self._terminated = True
        log_events.TerminatedTraining(model_name=self.model.name,
                                      source=self.name,
                                      epoch=self.model.epoch,
                                      reason=reason)
        return

    def train(self: Self, num_epochs: int) -> None:
        """
        Train the module for the specified number of epochs.

        Args:
            num_epochs: the number of epochs for which train the module.
        """
        if self.terminated:
            warnings.warn(exceptions.TerminatedTrainingWarning())
            return
        final_epoch = self.model.epoch + num_epochs
        log_events.StartTraining(self.model.name, self.model.epoch, final_epoch)
        for _ in range(num_epochs):
            log_events.StartEpoch(self.model.epoch + 1, final_epoch)
            self.pre_epoch_hooks.execute(self)
            self.__call__()
            self.post_epoch_hooks.execute(self)
            log_events.EndEpoch()
            if self.terminated:
                break
        log_events.EndTraining()
        return

    def train_until(self: Self, epoch: int) -> None:
        """
        Train the module until the specified epoch.

        Args:
            epoch: the final epoch in the training.

        """
        remaining_epochs = epoch - self.model.epoch
        if remaining_epochs > 0:
            self.train(remaining_epochs)
        if remaining_epochs < 0:
            warnings.warn(exceptions.PastEpochWarning(epoch, self.model.epoch))
        return

    def update_learning_rate(
            self,
            base_lr: Optional[float | dict[str, float]] = None,
            scheduler: Optional[p.SchedulerProtocol] = None,
    ) -> None:
        """
        It updates the learning rates for each parameters' group in the
        optimizer based on input learning rate(s) and scheduler.

        Args:
            base_lr: initial learning rates for named parameters or global
                value. Default keeps the original learning rates.
            scheduler: scheduler for the learning rates. Default keeps the
                original scheduler.
        """
        scheduler_name = None if scheduler is None else repr(scheduler)
        log_events.UpdateLearningRate(model_name=self.model.name,
                                      source=self.name,
                                      epoch=self.model.epoch,
                                      base_lr=base_lr,
                                      scheduler_name=scheduler_name)

        self._model_optimizer.update_learning_rate(base_lr, scheduler)

    @override
    def _run_backwards(self, outputs: _Output, targets: _Target) -> None:
        loss_value = self.calculator.forward(outputs, targets)
        try:
            if torch.isinf(loss_value) or torch.isnan(loss_value):
                raise exceptions.ConvergenceError(loss_value.item())
        except RuntimeError as re:
            if loss_value.numel() != 1:
                raise exceptions.LossNotScalarError(loss_value.shape)
            raise re
        self._scaler.scale(loss_value).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad()

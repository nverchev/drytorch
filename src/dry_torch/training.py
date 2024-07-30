import logging
from typing import Callable, Optional, Self, TypeVar

import torch
from torch.cuda import amp

from dry_torch import descriptors
from dry_torch import io
from dry_torch import exceptions
from dry_torch import tracking
from dry_torch import learning
from dry_torch import protocols as p
from dry_torch import log_settings
from dry_torch import evaluating
from dry_torch import binding
from dry_torch import hooks

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)

logger = logging.getLogger('dry_torch')


class Trainer(
    evaluating.Evaluation[_Input, _Target, _Output],
    p.TrainerProtocol,
):
    partition = descriptors.Split.TRAIN
    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        tracker: contain the module and the optimizing strategy.
        loader: dictionary with loaders for the training, and optionally,
         the validation and test datasets.
        loss_calc: the _loss_calc function, which needs to return batched values
         as in LossAndMetricsProtocol.
        mixed_precision: whether to use mixed precision computing.
         Optional, default to False.

    Methods:
        train:
        run the training session,
        optionally quickly evaluate on the validation dataset.

        hook_before_training_epoch:
        property for adding a hook before running the training session.
        hook_after_training_epoch:
        property for adding a hook after running the training session.
    """

    @binding.bind_to_model
    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            learning_scheme: learning.LearningScheme,
            loss_calc: p.LossCalculatorProtocol[_Output, _Target],
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            val_loader: Optional[
                p.LoaderProtocol[tuple[_Input, _Target]]
            ] = None,
            mixed_precision: bool = False,
    ) -> None:
        super().__init__(model,
                         loader=loader,
                         metrics_calc=loss_calc,
                         mixed_precision=mixed_precision)
        self._early_termination = False

        self._model_optimizer = learning.ModelOptimizer(model, learning_scheme)
        self._optimizer = self._model_optimizer.optimizer
        self._checkpoint = io.CheckpointIO(model, self._optimizer)
        self._scaler = amp.GradScaler(enabled=self._mixed_precision)
        self.pre_epoch_hooks = hooks.HookRegistry[Self]()
        self.post_epoch_hooks = hooks.HookRegistry[Self]()
        self._val_loader = val_loader
        self._validation = self._set_validation()
        self._early_termination = False
        return

    @property
    def model_tracking(self) -> tracking.ModelTracker:
        return tracking.Experiment.current().tracker[self.model.name]

    def _set_validation(self) -> Callable[[], None]:

        if self._val_loader is None:
            def validation() -> None:
                return
        else:
            validation = evaluating.Validation(self.model,
                                               loader=self._val_loader,
                                               metrics_calc=self._calculator)
            self.post_epoch_hooks.register(hooks.validation_hook)
        return validation

    def validate(self) -> None:
        return self._validation()

    def terminate_training(self) -> None:
        try:
            binding.unbind(self, self.model)
        except exceptions.NotBoundedError:
            pass
        self._early_termination = True
        return

    def __call__(self) -> None:
        self.train(1)
        return

    def train(self: Self, num_epochs: int) -> None:
        """
        Train the module for the specified number of epochs.

        Parameters:
            num_epochs: the number of epochs for which train the module.
        """
        logger.log(log_settings.INFO_LEVELS.training,
                   'Training %(model_name)s.',
                   {'model_name': self.model.name})
        if self._early_termination:
            logger.warning('Attempted to train module after termination.')
        for _ in range(num_epochs):
            if self._early_termination:
                return
            self.pre_epoch_hooks.execute(self)
            self.model_tracking.epoch += 1
            epoch_msg = '====> Epoch %(epoch)4d:'
            logger.log(log_settings.INFO_LEVELS.epoch,
                       epoch_msg, {'epoch': self.model_tracking.epoch})
            self._model_optimizer.update_learning_rate()
            self.model.module.train()
            try:
                self._run_epoch()
            except exceptions.ConvergenceError as ce:
                logger.error(ce, exc_info=True)
            self.post_epoch_hooks.execute(self)
        logger.log(log_settings.INFO_LEVELS.training, 'End of training.')
        return

    def _run_batch(self, inputs: _Input, targets: _Target) -> None:
        super()._run_batch(inputs, targets)
        self._calculator: p.LossCalculatorProtocol
        criterion = self._calculator.criterion.mean(0)
        try:
            self._scaler.scale(criterion).backward()
        except RuntimeError as re:
            if criterion.numel != 1:
                raise exceptions.NotBatchedError(list(criterion.shape))
            raise re
        except ValueError as ve:
            if torch.isinf(criterion) or torch.isnan(criterion):
                raise exceptions.ConvergenceError(criterion.item())
            raise ve
        self._loader.send(criterion.item())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad()

    def save_checkpoint(self, replace_previous: bool = False) -> None:
        self._checkpoint.save(replace_previous)

    def load_checkpoint(self, epoch: int = -1) -> None:
        self._checkpoint.load(epoch=epoch)

    def update_learning_rate(
            self, learning_rate: float | dict[str, float],
    ) -> None:
        self._model_optimizer.update_learning_rate(learning_rate)

    def __str__(self) -> str:
        return f'Trainer for {self.model.name}.'

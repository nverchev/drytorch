import logging
from typing import Callable, Self, TypeVar

import torch
from torch.cuda import amp
from typing_extensions import override

from . import descriptors
from . import io
from . import exceptions
from . import tracking
from . import learning
from . import protocols as p
from . import log_settings
from . import evaluating
from . import hooks
from . import registering

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

    @registering.register_kwargs
    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            learning_scheme: p.LearningProtocol,
            loss_calc: p.LossCalculatorProtocol[_Output, _Target],
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            mixed_precision: bool = False,
            name: str = '',
    ) -> None:
        super().__init__(model,
                         loader=loader,
                         metrics_calc=loss_calc,
                         mixed_precision=mixed_precision,
                         name=name)
        self._early_termination = False

        self._model_optimizer = learning.ModelOptimizer(model, learning_scheme)
        self._optimizer = self._model_optimizer.optimizer
        self._checkpoint = io.CheckpointIO(model, self._optimizer)
        self._scaler = amp.GradScaler(enabled=self._mixed_precision)
        self.pre_epoch_hooks = hooks.HookRegistry[Self]()
        self.post_epoch_hooks = hooks.HookRegistry[Self]()
        self._early_termination = False
        return

    @property
    def model_tracker(self) -> tracking.ModelTracker:
        return tracking.Experiment.current().tracker[self.model.name]

    def add_validation(
            self,
            val_loader: p.LoaderProtocol[tuple[_Input, _Target]]
    ) -> Callable[[], None]:

        validation = evaluating.Validation(self.model,
                                           loader=val_loader,
                                           metrics_calc=self._calculator)
        self.post_epoch_hooks.register(hooks.static_hook(validation))
        return validation

    @override
    def terminate_training(self) -> None:
        self._early_termination = True
        return

    @override
    def __call__(self, store_outputs: bool = False) -> None:
        self.model_tracker.epoch += 1
        epoch_msg = '====> Epoch %(epoch)4d:'
        logger.log(log_settings.INFO_LEVELS.epoch,
                   epoch_msg, {'epoch': self.model_tracker.epoch})
        self._model_optimizer.update_learning_rate()
        self.model.module.train()
        try:
            self._run_epoch(store_outputs)
        except exceptions.ConvergenceError as ce:
            logger.error(ce)
            self.terminate_training()
        return

    @override
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
            self.__call__()
            self.post_epoch_hooks.execute(self)
        logger.log(log_settings.INFO_LEVELS.training, 'End of training.')
        return

    def _run_batch(self,
                   inputs: _Input,
                   targets: _Target,
                   store_outputs: bool) -> None:
        super()._run_batch(inputs, targets, store_outputs=store_outputs)
        self._calculator: p.LossCalculatorProtocol
        criterion = self._calculator.criterion.mean(0)
        if torch.isinf(criterion) or torch.isnan(criterion):
            raise exceptions.ConvergenceError(criterion.item())
        try:
            self._scaler.scale(criterion).backward()
        except RuntimeError as re:
            if criterion.numel != 1:
                raise exceptions.NotBatchedError(list(criterion.shape))
            raise re
        self._loader.send(criterion.item())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad()

    @override
    def save_checkpoint(self, replace_previous: bool = False) -> None:
        self._checkpoint.save(replace_previous)

    @override
    def load_checkpoint(self, epoch: int = -1) -> None:
        self._checkpoint.load(epoch=epoch)

    @override
    def update_learning_rate(
            self, learning_rate: float | dict[str, float],
    ) -> None:
        self._model_optimizer.update_learning_rate(learning_rate)

    def __str__(self) -> str:
        return f'Trainer for {self.model.name}.'

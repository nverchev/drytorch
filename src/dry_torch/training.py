from __future__ import annotations
import logging

from typing import Callable, Optional, Self, TypeVar
import torch
from torch.cuda import amp

from dry_torch import checkpoint
from dry_torch import exceptions
from dry_torch import tracking
from dry_torch import model_utils
from dry_torch import protocols
from dry_torch import data_types
from dry_torch import default_logging
from dry_torch import loading
from dry_torch import testing

_Input = TypeVar('_Input', bound=data_types.InputType)
_Target = TypeVar('_Target', bound=data_types.TargetType)
_Output = TypeVar('_Output', bound=data_types.OutputType)

logger = logging.getLogger('dry_torch')


class Trainer(
    testing.Evaluator[_Input, _Target, _Output],
    protocols.TrainerProtocol,
):
    partition = data_types.Split.TRAIN
    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        model: contain the module and the optimizing strategy.
        train_loader: dictionary with loaders for the training, and optionally,
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

    @model_utils.log_kwargs(bind_to_model=True)
    def __init__(
            self,
            model: protocols.ModelProtocol[_Input, _Output],
            /,
            *,
            learning_scheme: protocols.LearningProtocol,
            loss_calc: protocols.LossCallable[_Output, _Target],
            train_loader: protocols.LoaderProtocol[_Input, _Target],
            val_loader: Optional[
                protocols.LoaderProtocol[_Input, _Target]
            ] = None,
            mixed_precision: bool = False,
    ) -> None:
        super().__init__(model, loader=train_loader)
        self.model = model
        self.model_optimizer = model_utils.ModelOptimizer(model,
                                                          learning_scheme)
        self._loader = loading.TqdmLoader[_Input, _Target](train_loader)
        device_is_cuda = self.model.device.type == 'cuda'
        enable_mixed_precision = mixed_precision and device_is_cuda
        self._scaler = amp.GradScaler(enabled=enable_mixed_precision)
        self._mixed_precision = mixed_precision

        self._loss_calc = loss_calc
        self._pre_epoch_hooks: list[Callable[[Self], None]] = []
        self._post_epoch_hooks: list[Callable[[Self], None]] = []

        if val_loader is None:
            self.validate: Callable[[], None] = lambda: None
        else:
            self.validate = testing.Validator(
                model,
                loader=val_loader,
                metrics_calc=self._loss_calc.metrics_calc
            )
            self._activate_validation()

        self.early_termination = False
        self.checkpoint = checkpoint.CheckpointIO(
            model,
            self.model_optimizer.optimizer,
        )
        return

    @property
    def model_tracking(self) -> tracking.ModelTracking:
        return tracking.Experiment.current().model[self.model.name]

    def _activate_validation(self: Self) -> None:
        def validate(instance: Self) -> None:
            instance.validate()
            return

        self._post_epoch_hooks.append(validate)
        return

    def terminate_training(self) -> None:
        model_utils.unbind(self, self.model)
        self.early_termination = True
        return

    def train(self, num_epochs: int) -> None:
        """
        Train the module for the specified number of epochs.

        Parameters:
            num_epochs: the number of epochs for which train the module.
        """
        logger.log(default_logging.INFO_LEVELS.training,
                   'Training %(model_name)s.',
                   {'model_name': self.model.name})
        if self.early_termination:
            logger.warning('Attempted to train module after termination.')
        for _ in range(num_epochs):
            self.model_tracking.epoch += 1
            epoch_msg = '====> Epoch %(epoch)4d:'
            logger.log(default_logging.INFO_LEVELS.epoch,
                       epoch_msg, {'epoch': self.model_tracking.epoch})
            try:
                self._train()
            except exceptions.ConvergenceError as ce:
                logger.error(ce)

        logger.log(default_logging.INFO_LEVELS.training, 'End of training.')
        return

    def _train(self) -> None:
        """
        Train the module for the specified number of epochs.
        """
        if self.early_termination:
            return
        self.model_optimizer.update_learning_rate()
        self.model.module.train()
        self.exec_pre_epoch_hooks()
        self._run_epoch()
        self.exec_post_epoch_hooks()
        return

    def _run_batch(self, batch: tuple[_Input, _Target]) -> None:
        inputs, targets = batch
        with torch.autocast(device_type=self.model.device.type,
                            enabled=self._mixed_precision):
            outputs = self.model(inputs)
            batched_performance = self._loss_calc(outputs, targets)
        criterion: torch.Tensor = batched_performance.criterion
        self._loader.send({'Loss': criterion.item()})
        try:
            self._scaler.scale(criterion).backward()
        except ValueError as ve:
            if torch.isinf(criterion) or torch.isnan(criterion):
                raise exceptions.ConvergenceError(criterion.item())
            raise ve
        self._scaler.step(self.model_optimizer.optimizer)
        self._scaler.update()
        self.model_optimizer.optimizer.zero_grad()
        self.register_metrics(batched_performance.metrics)

    def save_checkpoint(self) -> None:
        self.checkpoint.save()

    def load_checkpoint(self, epoch=-1) -> None:
        self.checkpoint.load(epoch=epoch)

    def add_pre_epoch_hook(
            self: Self,
            hook: Callable[[Self], None]
    ) -> None:
        self._pre_epoch_hooks.append(hook)
        return

    def add_post_epoch_hook(
            self: Self,
            hook: Callable[[Self], None]
    ) -> None:
        self._post_epoch_hooks.append(hook)
        return

    def exec_pre_epoch_hooks(self: Self) -> None:
        """
        This hook is called before running the training session.
        """
        for hook in self._pre_epoch_hooks:
            hook(self)
        return

    def exec_post_epoch_hooks(self: Self) -> None:
        """
        This hook is called before running the training session.
        """
        for hook in self._post_epoch_hooks:
            hook(self)
        return

    def __str__(self) -> str:
        return f'Trainer for {self.model.name}.'

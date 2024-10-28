import warnings
from typing import Self, TypeVar, Optional

import torch
from torch.cuda import amp
from typing_extensions import override

from src.dry_torch import descriptors
from src.dry_torch import checkpoint
from src.dry_torch import exceptions
from src.dry_torch import learning
from src.dry_torch import protocols as p
from src.dry_torch import events
from src.dry_torch import evaluating
from src.dry_torch import hooks
from src.dry_torch import registering

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


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
        self._terminated = False

        self._model_optimizer = learning.ModelOptimizer(model, learning_scheme)
        self._optimizer = self._model_optimizer.optimizer
        self._checkpoint = checkpoint.CheckpointIO(model, self._optimizer)
        self._scaler = amp.GradScaler(enabled=self._mixed_precision)
        self.validation: Optional[evaluating.Validation] = None
        self.pre_epoch_hooks = hooks.HookRegistry[Self]()
        self.post_epoch_hooks = hooks.HookRegistry[Self]()
        self._terminated = False
        return

    def add_validation(
            self,
            val_loader: p.LoaderProtocol[tuple[_Input, _Target]]
    ) -> None:

        validation = evaluating.Validation(self.model,
                                           loader=val_loader,
                                           metrics_calc=self._calculator)
        self.post_epoch_hooks.register(hooks.static_hook(validation))
        self.validation = validation
        return

    @property
    def terminated(self) -> bool:
        return self._terminated

    def terminate_training(self) -> None:
        self._terminated = True
        return

    @override
    def __call__(self, store_outputs: bool = False) -> None:
        self.model.increment_epoch()
        self._model_optimizer.update_learning_rate()
        self.model.module.train()
        try:
            self._run_epoch(store_outputs)
        except exceptions.ConvergenceError as ce:
            events.ModelDidNotConverge(ce)
            self.terminate_training()

        return

    def train(self: Self, num_epochs: int) -> None:
        """
        Train the module for the specified number of epochs.

        Parameters:
            num_epochs: the number of epochs for which train the module.
        """
        if self._terminated:
            warnings.warn('Attempted to train module after termination.')
            return
        final_epoch = self.model.epoch + num_epochs
        events.StartTraining(self.model.name, self.model.epoch, final_epoch)
        for _ in range(num_epochs):
            if self._terminated:
                break
            events.StartEpoch(self.model.epoch + 1, final_epoch)
            self.pre_epoch_hooks.execute(self)
            self.__call__()
            self.post_epoch_hooks.execute(self)
            events.EndEpoch()
        events.EndTraining()
        return

    def train_until(self: Self, epoch: int) -> None:
        """
        Train the module until the specified epoch.

        Parameters:
            epoch: the final epoch in the training.

        """
        remaining_epochs = epoch - self.model.epoch
        if remaining_epochs > 0:
            self.train(remaining_epochs)
        if remaining_epochs < 0:
            warnings.warn(exceptions.PastEpochWarning(epoch, self.model.epoch))
        return

    def _run_backward(self) -> None:
        self._calculator: p.LossCalculatorProtocol
        criterion = self._calculator.criterion.mean(0)
        if torch.isinf(criterion) or torch.isnan(criterion):
            raise exceptions.ConvergenceError(criterion.item())
        try:
            self._scaler.scale(criterion).backward()
        except RuntimeError as re:
            if criterion.numel != 1:
                raise exceptions.MetricsNotAVectorError(list(criterion.shape))
            raise re
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

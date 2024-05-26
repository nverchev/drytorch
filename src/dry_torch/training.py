from __future__ import annotations

import sys
import logging

from collections import defaultdict
from typing import Callable, Optional, Self, TypeVar, Generic
import pandas as pd
import torch
from torch.cuda import amp

from dry_torch import exceptions
from dry_torch import tracking
from dry_torch import model_binding
from dry_torch import structures
from dry_torch import recursive_ops
from dry_torch import protocols
from dry_torch import data_types
from dry_torch import default_logging
from dry_torch import loading

_Input = TypeVar('_Input', bound=data_types.InputType)
_Target = TypeVar('_Target', bound=data_types.TargetType)
_Output = TypeVar('_Output', bound=data_types.OutputType)

logger = logging.getLogger('dry_torch')


class Test:
    max_stored_output: int = sys.maxsize

    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        model_optimizer: contain the model and the optimizing strategy.
        loaders: dictionary with loaders for the training, and optionally,
         the validation and test datasets.
        loss_calc: the _loss_calc function, which needs to return batched values
         as in LossAndMetricsProtocol.
        metrics_calc: the test metrics function, returning TestMetricsProtocol.
         If None, _loss_calc will be used instead.
        mixed_precision: whether to use mixed precision computing.
         Optional, default to False.

    Attributes:
        max_stored_output:
        the maximum number of outputs to store when testing.
        update_frequency:
        number of times the progress bar updates in one epoch.
        test_outputs:
        An instance of TorchDictList that stores the last test evaluation.

    Methods:
        train:
        run the training session,
        optionally quickly evaluate on the validation dataset.
        test: evaluate on the specified partition of the dataset.
        hook_before_training_epoch:
        property for adding a hook before running the training session.
        hook_after_training_epoch:
        property for adding a hook after running the training session.
    """

    def __init__(
            self,
            model_optimizer: protocols.ModelOptimizerProtocol[_Input, _Target],
            /,
            *,
            loader: protocols.LoaderProtocol[_Input, _Target],
            metrics_calc: protocols.MetricsCallable[_Output, _Target]
    ) -> None:

        self._model_optimizer = model_optimizer
        self._loader = loading.TqdmLoader[_Input, _Target](loader)
        self._metrics_calc = metrics_calc
        self.test_outputs = structures.TorchDictList()
        return

    @property
    def model_info(self) -> tracking.ModelTracking:
        exp = tracking.Experiment.get_active_environment()
        return exp.model[self._model_optimizer.name]

    @torch.inference_mode()
    def __call__(self,
                 partition: data_types.Split = data_types.Split.VAL,
                 save_outputs: bool = False) -> None:
        """
        Evaluates the model's performance on the specified partition of the
        dataset.

        Parameters:
            partition: The partition of the dataset on which to evaluate the
            model's performance.  Default to 'val'.
            save_outputs: if the flag is active store the model outputs in the
            test_outputs attribute. Default to False.

        """
        if save_outputs:
            self.test_outputs.clear()
        self._model_optimizer.model.eval()
        self._run_epoch(partition=partition,
                        save_outputs=save_outputs,
                        use_test_metrics=True)
        return

    def _run_epoch(self,
                   partition: data_types.Split,
                   save_outputs: bool = False) -> None:
        """
           Run a single epoch of training or evaluation.

           Parameters:
               save_outputs: if the flag is active, store the model outputs.
                Default to False.

           """
        num_batch: int = len(self._loader)
        partition_log: pd.DataFrame = self.model_info.log[partition]
        epoch_log: defaultdict[str, float] = defaultdict(float)

        for inputs, targets in self._loader:
            batched_perf, outputs = self._run_batch(inputs, targets)
            # if you get a memory error, decrease max_stored_output
            # if save_outputs and self.max_stored_output >= epoch_seen:
            #     self.test_outputs.extend(
            #         structures.TorchDictList.from_batch(outputs)
            #     )
            for metric, batched_value in batched_perf.metrics.items():
                epoch_log[metric] += batched_value.sum(0).item()
                #self._tqdm_loader.send({'Loss': batched_perf.criterion.mean(0).item()}

        inference_flag = torch.is_inference_mode_enabled()
        log_msg_list: list[str] = ['Average %(split)s metric(s):']
        log_args: dict[str, str | float] = {'split': partition.name.lower()}
        for metric, value in epoch_log.items():
            value /= self._loader.loader.dataset_len
            if inference_flag ^ (partition == data_types.Split.TRAIN):
                # Evaluating train does not overwrite log
                partition_log.loc[self.model_info.epoch, metric] = value
            log_msg_list.append(f'%({metric})s: %({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(default_logging.INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        return

    def _run_batch(
            self,
            inputs: _Input,
            targets: _Target,
    ) -> tuple[protocols.MetricsProtocol, _Output]:
        inputs, targets = (
            recursive_ops.recursive_to(
                [inputs, targets], self._model_optimizer.device
            )
        )
        outputs = self._model_optimizer(inputs)
        batched_performance = self._metrics_calc(outputs, targets)
        return batched_performance, outputs

    def __str__(self) -> str:
        return f'Trainer for {self._model_optimizer.name}.'


class Trainer(Generic[_Input, _Target, _Output], protocols.TrainerProtocol):
    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        model_optimizer: contain the model and the optimizing strategy.
        loaders: dictionary with loaders for the training, and optionally,
         the validation and test datasets.
        loss_calc: the _loss_calc function, which needs to return batched values
         as in LossAndMetricsProtocol.
        metrics_calc: the test metrics function, returning TestMetricsProtocol.
         If None, _loss_calc will be used instead.
        mixed_precision: whether to use mixed precision computing.
         Optional, default to False.

    Attributes:
        max_stored_output:
        the maximum number of outputs to store when testing.
        tqdm_update_frequency:
        number of times the progress bar updates in one epoch.
        test_outputs:
        An instance of TorchDictList that stores the last test evaluation.

    Methods:
        train:
        run the training session,
        optionally quickly evaluate on the validation dataset.
        test: evaluate on the specified partition of the dataset.
        hook_before_training_epoch:
        property for adding a hook before running the training session.
        hook_after_training_epoch:
        property for adding a hook after running the training session.
    """

    @model_binding.bind_to_model
    def __init__(
            self,
            model_optimizer: protocols.ModelOptimizerProtocol[_Input, _Output],
            /,
            *,
            loader: protocols.LoaderProtocol[_Input, _Target],
            loss_calc: protocols.LossCallable[_Output, _Target],
            mixed_precision: bool = False,
    ) -> None:

        self._model_optimizer = model_optimizer
        self._loader = loader
        self._tqdm_loader = loading.TqdmLoader[_Input, _Target](loader)
        device_is_cuda = model_optimizer.device.type == 'cuda'
        enable_mixed_precision = mixed_precision and device_is_cuda
        self._scaler = amp.GradScaler(enabled=enable_mixed_precision)
        self._mixed_precision = mixed_precision

        self._loss_calc = loss_calc
        self._pre_epoch_hooks: list[
            Callable[[protocols.TrainerProtocol], None]] = []
        self._post_epoch_hooks: list[
            Callable[[protocols.TrainerProtocol], None]] = []
        self._post_epoch_hooks.append(validation_hook)

        display_epoch_info = logger.level > default_logging.INFO_LEVELS.epoch
        self.disable_bar = display_epoch_info and sys.stdout.isatty()

        self.early_termination = False
        return

    @property
    def model_info(self) -> tracking.ModelTracking:
        exp = tracking.Experiment.get_active_environment()
        return exp.model[self._model_optimizer.name]

    def terminate_training(self) -> None:
        self.early_termination = True

    def train(self, num_epoch: int, val_after_train: bool = False) -> None:
        """
        Train the model for the specified number of epochs.

        Parameters:
            num_epoch: the number of epochs for which train the model.
            val_after_train: if the flag is active, evaluate loss function
            on the validation dataset. Default to False.
        """
        logger.log(default_logging.INFO_LEVELS.training,
                   'Training %(model_name)s:',
                   {'model_name': self._model_optimizer.name})

        if self.early_termination:
            logger.warning('Training has been terminated.')
        for _ in range(num_epoch):
            if self.early_termination:
                return
            self._model_optimizer.update_learning_rate()
            self.model_info.epoch += 1

            # Logging
            epoch_msg = '====> Epoch %(epoch)4d:' + ' ...' * self.disable_bar
            logger.log(default_logging.INFO_LEVELS.epoch,
                       epoch_msg, {'epoch': self.model_info.epoch})

            self._model_optimizer.model.train()
            self.exec_pre_epoch_hooks()
            try:
                self._run_epoch(partition=data_types.Split.TRAIN)
            except exceptions.ConvergenceError as ce:

                # Logging
                logger.error(ce)
                self.terminate_training()

            self.exec_post_epoch_hooks()
        return

    def validate(self):
        self._model_optimizer.model.eval()
        with torch.inference_mode():
            self._run_epoch(partition=data_types.Split.VAL)

    def _run_epoch(self,
                   partition: data_types.Split,
                   save_outputs: bool = False) -> None:
        """
           Run a single epoch of training or evaluation.

           Parameters:
               partition: The partition of the dataset on which to evaluate
                the model's performance.
               save_outputs: if the flag is active, store the model outputs.
                Default to False.

           """
        partition_log: pd.DataFrame = self.model_info.log[partition]
        epoch_log: defaultdict[str, float] = defaultdict(float)

        for inputs, targets in self._tqdm_loader:
            batched_perf = self._run_batch(inputs, targets)

            # Averaging
            for metric, batched_value in batched_perf.metrics.items():
                epoch_log[metric] += batched_value.sum(0).item()

        log_msg_list: list[str] = ['Average %(split)s metric(s):']
        log_args: dict[str, str | float] = {'split': partition.name.lower()}

        for metric, value in epoch_log.items():
            # Averaging
            value /= self._tqdm_loader.dataset_len

            # Logging
            partition_log.loc[self.model_info.epoch, metric] = value
            log_msg_list.append(f'%({metric})s: %({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(default_logging.INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        return

    def _run_batch(self, inputs: _Input, targets: _Target) -> (
            protocols.LossAndMetricsProtocol):
        inputs, targets = (
            recursive_ops.recursive_to(
                [inputs, targets], self._model_optimizer.device
            )
        )
        with torch.autocast(device_type=self._model_optimizer.device.type,
                            enabled=self._mixed_precision):
            outputs = self._model_optimizer(inputs)
            batched_performance = self._loss_calc(outputs, targets)
            criterion: torch.Tensor = batched_performance.criterion.mean(0)
        self._tqdm_loader.send({'Loss': criterion.item()})
        if not torch.is_inference_mode_enabled():
            try:
                self._scaler.scale(criterion).backward()
            except ValueError as ve:
                if torch.isinf(criterion) or torch.isnan(criterion):
                    raise exceptions.ConvergenceError(criterion.item())
                raise ve
            self._scaler.step(self._model_optimizer.optimizer)
            self._scaler.update()
            self._model_optimizer.optimizer.zero_grad()
        return batched_performance

    def add_pre_epoch_hook(
            self,
            hook: Callable[[protocols.TrainerProtocol], None]
    ) -> None:
        self._pre_epoch_hooks.append(hook)
        return

    def add_post_epoch_hook(
            self,
            hook: Callable[[protocols.TrainerProtocol], None]
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
        return f'Trainer for {self._model_optimizer.name}.'


def validation_hook(trainer: protocols.TrainerProtocol) -> None:
    return trainer.validate()

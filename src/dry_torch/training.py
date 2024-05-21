from __future__ import annotations

import sys
import logging

from collections import defaultdict
from typing import Callable, Optional, Self, TypeVar
import pandas as pd
import torch
from torch.cuda import amp
from tqdm.auto import tqdm

from dry_torch import exceptions
from dry_torch import tracking
from dry_torch import model_binding
from dry_torch import structures
from dry_torch import recursive_ops
from dry_torch import protocols
from dry_torch import data_types
from dry_torch import default_logging

_Input = TypeVar('_Input', bound=data_types.InputType)
_Target = TypeVar('_Target', bound=data_types.TargetType)
_Output = TypeVar('_Output', bound=data_types.OutputType)

logger = logging.getLogger('dry_torch')


class Trainer(protocols.TrainerProtocol[_Input, _Target, _Output]):
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
    tqdm_update_frequency = 10
    max_stored_output: int = sys.maxsize

    @model_binding.bind_to_model
    def __init__(
            self,
            model_optimizer: (
                    protocols.ModelOptimizerProtocol[_Input, _Output]
            ),
            /,
            *,
            loaders: protocols.LoadersProtocol[_Input, _Target],
            loss_calc: protocols.LossCallable[_Output, _Target],
            metrics_calc: (
                    Optional[protocols.MetricsCallable[_Output, _Target]]
            ) = None,
            mixed_precision: bool = False,
    ) -> None:

        self._model_optimizer = model_optimizer
        self._data_loaders = loaders
        device_is_cuda = model_optimizer.device.type == 'cuda'
        enable_mixed_precision = mixed_precision and device_is_cuda
        self._scaler = amp.GradScaler(enabled=enable_mixed_precision)
        self._mixed_precision = mixed_precision

        self._loss_calc = loss_calc
        self._metrics_calc = loss_calc if metrics_calc is None else metrics_calc

        self._hooks_before_training_epoch: list[Callable[[Self], None]] = []
        self._hooks_after_training_epoch: list[Callable[[Self], None]] = []
        self.test_outputs = structures.TorchDictList()

        display_epoch_info = logger.level > default_logging.INFO_LEVELS.epoch
        self.disable_bar = display_epoch_info and sys.stdout.isatty()
        return

    @property
    def model_info(self) -> tracking.ModelTracking:
        exp = tracking.Experiment.get_active_environment()
        return exp.model[self._model_optimizer.name]

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
        for _ in range(num_epoch):
            self._model_optimizer.update_learning_rate()
            self.model_info.epoch += 1
            epoch_msg = '====> Epoch %(epoch)4d:' + ' ...' * self.disable_bar
            logger.log(default_logging.INFO_LEVELS.epoch,
                       epoch_msg, {'epoch': self.model_info.epoch})
            self._model_optimizer.model.train()
            self.exec_hooks_before_training_epoch()
            try:
                self._run_epoch(partition=data_types.Split.TRAIN)
            except exceptions.ConvergenceError as ce:
                logger.error(ce)
                return
            self.exec_hooks_after_training_epoch()
            if val_after_train:  # check losses on val
                self._model_optimizer.model.eval()
                with torch.inference_mode():
                    self._run_epoch(partition=data_types.Split.VAL,
                                    use_test_metrics=False)
        return

    @torch.inference_mode()
    def test(self,
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
        self._model_optimizer.model.eval()
        self._run_epoch(partition=partition,
                        save_outputs=save_outputs,
                        use_test_metrics=True)
        print()
        return

    def _run_epoch(self,
                   partition: data_types.Split,
                   save_outputs: bool = False,
                   use_test_metrics: bool = False) -> None:
        """
           Run a single epoch of training or evaluation.

           Parameters:
               partition: The partition of the dataset on which to evaluate
                the model's performance.
               save_outputs: if the flag is active, store the model outputs.
                Default to False.
               use_test_metrics: if the flag is active use the metrics function
               instead of the loss function.

           """
        loader = self._data_loaders.get_loader(partition)
        num_batch: int = len(loader)
        dataset_length: int = self._data_loaders.datasets_length[partition]
        if use_test_metrics and not torch.is_inference_mode_enabled():
            msg = 'Cannot use test metrics when inference mode is not enabled.'
            raise ValueError(msg)
        if save_outputs:
            self.test_outputs.clear()
        partition_log: pd.DataFrame = self.model_info.log[partition]
        epoch_log: defaultdict[str, float] = defaultdict(float)

        with tqdm(enumerate(loader),
                  total=num_batch,
                  disable=self.disable_bar,
                  file=sys.stdout) as tqdm_loader:
            epoch_seen: int = 0
            batch_data: tuple[int, tuple[_Input, _Target]]
            for batch_data in tqdm_loader:
                (batch_idx, (inputs, targets)) = batch_data
                epoch_seen += self._data_loaders.batch_size
                batched_perf, outputs = self._run_batch(inputs,
                                                        targets,
                                                        use_test_metrics)
                # if you get a memory error, decrease max_stored_output
                if save_outputs and self.max_stored_output >= epoch_seen:
                    self.test_outputs.extend(
                        structures.TorchDictList.from_batch(outputs)
                    )
                for metric, batched_value in batched_perf.metrics.items():
                    epoch_log[metric] += batched_value.sum(0).item()

                update_interval = num_batch // self.tqdm_update_frequency or 1
                # or 1 needed for small batches
                if batch_idx % update_interval == 0:
                    monitor: dict[str, int | float] = {
                        'Seen': min(epoch_seen, dataset_length),
                    }
                    if hasattr(batched_perf, 'criterion'):
                        monitor['Loss'] = batched_perf.criterion.mean(0).item()
                    tqdm_loader.set_postfix(monitor)

        inference_flag = torch.is_inference_mode_enabled()
        log_msg_list: list[str] = ['Average %(split)s metric(s):']
        log_args: dict[str, str | float] = {'split': partition.name.lower()}
        for metric, value in epoch_log.items():
            value /= epoch_seen
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
            use_test_metrics: bool,
    ) -> tuple[protocols.MetricsProtocol, _Output]:
        inputs, targets = (
            recursive_ops.recursive_to(
                [inputs, targets], self._model_optimizer.device
            )
        )
        with torch.autocast(device_type=self._model_optimizer.device.type,
                            enabled=self._mixed_precision):
            outputs = self._model_optimizer(inputs)
            if use_test_metrics:
                batched_performance = self._metrics_calc(outputs, targets)
            else:
                batched_performance = self._loss_calc(outputs, targets)
                criterion: torch.Tensor = batched_performance.criterion.mean(0)
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
        return batched_performance, outputs

    def add_hook_before_training_epoch(
            self,
            hook: Callable[[Trainer[_Input, _Target, _Output]], None]
    ) -> None:
        return self._hooks_before_training_epoch.append(hook)

    def add_hook_after_training_epoch(
            self,
            hook: Callable[[Trainer[_Input, _Target, _Output]], None]
    ) -> None:
        return self._hooks_before_training_epoch.append(hook)

    def exec_hooks_before_training_epoch(self: Self) -> None:
        """
        This hook is called before running the training session.
        """
        for hook in self._hooks_before_training_epoch:
            hook(self)
        return

    def exec_hooks_after_training_epoch(self: Self) -> None:
        """
        This hook is called before running the training session.
        """
        for hook in self._hooks_after_training_epoch:
            hook(self)
        return

    def __str__(self) -> str:
        return f'Trainer for {self._model_optimizer.name}.'

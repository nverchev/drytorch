from __future__ import annotations

import sys
import logging
import abc
from typing import TypeVar, Generic, Optional
import pandas as pd
import torch

from dry_torch import tracking
from dry_torch import model_utils
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


class Evaluator(Generic[_Input, _Target, _Output], metaclass=abc.ABCMeta):
    partition: data_types.Split

    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        model: contain the module and the optimizing strategy.
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
        save_outputs: if the flag is active store the module outputs in the
            test_outputs attribute. Default to False.

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
            model: protocols.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: protocols.LoaderProtocol[_Input, _Target],
            metrics_calc: protocols.MetricsCallable[_Output, _Target],
            save_outputs: bool = False,

    ) -> None:
        self.model = model
        self._loader = loading.TqdmLoader[_Input, _Target](loader)
        self._metrics_calc = metrics_calc
        self.test_outputs = structures.TorchDictList()
        self._metrics = structures.TorchAggregate()
        self.save_outputs = save_outputs

        return

    @property
    def model_tracking(self) -> tracking.ModelTracking:
        return tracking.Experiment.current().model[self.model.name]

    @property
    def metrics(self) -> dict[str, float]:
        out = self._metrics.reduce()
        return out

    @torch.inference_mode()
    def __call__(self) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        self.model.module.eval()
        self._run_epoch()
        return

    def register_metrics(self, metrics: dict[str, torch.Tensor]):
        self._metrics += metrics

    def log_metrics(self) -> None:

        log_msg_list: list[str] = ['Average %(split)s metric(s):']
        split_str = self.partition.name.lower()
        log_args: dict[str, str | float] = {'split': split_str}
        for metric, value in self.metrics.items():
            self.update_partition_log(metric, value)
            log_msg_list.append(f'%({metric})s: %({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(default_logging.INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        self._metrics = structures.TorchAggregate()
        return

    def _run_epoch(self):
        if self.save_outputs:
            self.test_outputs.clear()
        for batch in self._loader:
            batch = recursive_ops.recursive_to(batch, self.model.device)
            self._run_batch(batch)
        self.log_metrics()

    def _run_batch(self, batch: tuple[_Input, _Target]) -> None:
        inputs, targets = batch
        outputs = self.model(inputs)
        self.register_metrics(self._metrics_calc(outputs, targets))

    def update_partition_log(self, metric: str, value: float) -> None:
        return


class Validator(Evaluator[_Input, _Target, _Output]):
    partition = data_types.Split.VAL

    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        model: contain the module and the optimizing strategy.
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
        save_outputs: if the flag is active store the module outputs in the
            test_outputs attribute. Default to False.

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

    def update_partition_log(self, metric: str, value: float) -> None:
        partition_log = self.model_tracking.log[self.partition]
        partition_log.loc[self.model_tracking.epoch, metric] = value
        return

    def __str__(self) -> str:
        return f'Trainer for {self.model.name}.'


class Test(Evaluator[_Input, _Target, _Output]):
    partition = data_types.Split.TEST
    max_stored_output: int = sys.maxsize
    default_test_name = tracking.default_name('Test_')

    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        model: contain the module and the optimizing strategy.
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
        save_outputs: if the flag is active store the module outputs in the
            test_outputs attribute. Default to False.

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

    @model_utils.log_kwargs()
    def __init__(
            self,
            model: protocols.ModelProtocol[_Input, _Output],
            /,
            *,
            test_name: Optional[str] = None,
            test_loader: protocols.LoaderProtocol[_Input, _Target],
            metrics_calc: protocols.MetricsCallable[_Output, _Target],
    ) -> None:
        super().__init__(model, loader=test_loader, metrics_calc=metrics_calc)
        self.test_name = test_name or self.__class__.default_test_name()
        self.model = model
        return

    @torch.inference_mode()
    def __call__(self) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        logger.log(default_logging.INFO_LEVELS.experiment,
                   '%(test_name):',
                   {'test_name': self.test_name})
        super().__call__()
        return

    def _run_batch(self, batch: tuple[_Input, _Target]) -> None:
        inputs, targets = batch
        outputs = self.model(inputs)
        self.register_metrics(self._metrics_calc(outputs, targets))
        if self.save_outputs:
            self.test_outputs.extend(
                structures.TorchDictList.from_batch(outputs)
            )

    def log_metrics(self) -> None:

        log_msg_list: list[str] = ['Average %(split)s metric(s):']
        log_args: dict[str, str | float] = {
            'split': self.partition.name.lower()
        }

        partition_log: pd.DataFrame = self.model_tracking.log[
            self.partition
        ]
        for metric, value in self.metrics.items():
            partition_log.loc[self.model_tracking.epoch, metric] = value
            log_msg_list.append(f'%({metric})s: %({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(default_logging.INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        self._metrics = structures.TorchAggregate()
        return

    def __str__(self) -> str:
        return f'Trainer for {self.model.name}.'

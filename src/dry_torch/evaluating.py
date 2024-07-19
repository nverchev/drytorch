from __future__ import annotations

import sys
import logging
import abc
import warnings
from typing import TypeVar, Generic

import dry_torch.model_bindings
import dry_torch.protocols
import torch

from dry_torch import exceptions
from dry_torch import tracking
from dry_torch import io
from dry_torch import learning
from dry_torch import structures
from dry_torch import recursive_ops
from dry_torch import protocols as p
from dry_torch import default_logging
from dry_torch import loading

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)

logger = logging.getLogger('dry_torch')


class Evaluation(Generic[_Input, _Target, _Output], metaclass=abc.ABCMeta):
    max_stored_output: int = sys.maxsize
    partition: dry_torch.protocols.Split
    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        tracking: contain the module and the optimizing strategy.
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
        stored_outputs:
            An instance of TorchDictList that stores the last test evaluation.
        _store_outputs: if the flag is active store the module outputs in the
            stored_outputs attribute. Default to False.

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
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            metrics_calc: p.MetricsCalculatorProtocol[_Output, _Target],
            store_outputs: bool = False,
            mixed_precision: bool = False,
    ) -> None:
        self.model = model
        self._loader = loading.TqdmLoader[tuple[_Input, _Target]](loader)
        self._calculator = metrics_calc
        self._store_outputs = store_outputs
        device_is_cuda = self.model.device.type == 'cuda'
        self._mixed_precision = mixed_precision and device_is_cuda
        self._metrics = structures.TorchAggregate()
        self.stored_outputs = structures.NumpyDictList()
        return

    @property
    def model_tracking(self) -> tracking.ModelTracker:
        return tracking.Experiment.current().tracking[self.model.name]

    @property
    def partition_log(self):
        return self.model_tracking.log[self.partition]

    @property
    def metrics(self) -> dict[str, float]:
        out = self._metrics.reduce()
        return out

    @abc.abstractmethod
    def __call__(self) -> None:
        ...

    def log_metrics(self) -> None:

        log_msg_list: list[str] = ['%(desc)-24s']
        desc = f'Average {self.partition.name.lower()} metrics:'
        log_args: dict[str, str | float] = {'desc': desc}
        for metric, value in self.metrics.items():
            self._update_partition_log(metric, value)
            log_msg_list.append(f'%({metric})16s: %({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(default_logging.INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        self._metrics = structures.TorchAggregate()
        return

    def _run_epoch(self):
        if self._store_outputs:
            self.stored_outputs.clear()
        for batch in self._loader:
            batch = recursive_ops.recursive_to(batch, self.model.device)
            self._run_batch(*batch)
            self._calculator.reset_calculated()
        self.log_metrics()

    def _run_batch(self, inputs: _Input, targets: _Target) -> None:
        with torch.autocast(device_type=self.model.device.type,
                            enabled=self._mixed_precision):
            outputs = self.model(inputs)
            self._calculator.calculate(outputs, targets)
            if self._store_outputs:
                self._store(outputs)
        self._metrics += self._calculator.metrics

    def _store(self, outputs: _Output) -> None:
        try:
            dict_batch = structures.NumpyDictList.from_batch(outputs)
            self.stored_outputs.extend(dict_batch)
        except exceptions.NotATensorError as type_err:
            warnings.warn(exceptions.CannotStoreOutputWarning(str(type_err)))
        except exceptions.NoToDictMethodError as attr_err:
            warnings.warn(exceptions.CannotStoreOutputWarning(str(attr_err)))
        except exceptions.DifferentBatchSizeError as value_err:
            warnings.warn(exceptions.CannotStoreOutputWarning(str(value_err)))
        return

    def _update_partition_log(self, metric: str, value: float) -> None:
        self.partition_log.loc[self.model_tracking.epoch, metric] = value
        return

    def __str__(self) -> str:
        return f'Base Evaluator for {self.model.name}.'


class Diagnostic(Evaluation):
    partition = dry_torch.protocols.Split.TRAIN

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

    def _update_partition_log(self, metric: str, value: float) -> None:
        return


class Validation(Evaluation[_Input, _Target, _Output]):
    partition = dry_torch.protocols.Split.VAL

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

    def __str__(self) -> str:
        return f'Validator for {self.model.name}.'


class Test(Evaluation[_Input, _Target, _Output]):
    partition = dry_torch.protocols.Split.TEST

    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        tracking: contain the module and the optimizing strategy.
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
        stored_outputs:
            An instance of TorchDictList that stores the last test evaluation.
        _store_outputs: if the flag is active store the module outputs in the
            stored_outputs attribute. Default to False.

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

    @dry_torch.model_bindings.bind_to_model
    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            metrics_calc: p.MetricsCalculatorProtocol[_Output, _Target],
            name: str = '',
            store_outputs: bool = False,
    ) -> None:
        super().__init__(model,
                         loader=loader,
                         metrics_calc=metrics_calc,
                         store_outputs=store_outputs)
        self.test_name = name or self._get_default_name()
        self._checkpoint = io.LogIO(model.name)
        return

    def _get_default_name(self) -> str:
        return repr(self.model_tracking.bindings[self.__class__.__name__])

    @torch.inference_mode()
    def __call__(self) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        try:
            dry_torch.model_bindings.unbind(self, self.model)
        except exceptions.NotBoundedError:
            warnings.warn(exceptions.AlreadyTestedWarning())
            return

        logger.log(default_logging.INFO_LEVELS.experiment,
                   '%(test_name)s:',
                   {'test_name': self.test_name})
        self.model.module.eval()
        self._run_epoch()
        self._checkpoint.save()
        return

    def _update_partition_log(self, metric: str, value: float) -> None:
        self.partition_log.loc[self.test_name, metric] = value
        return

    def __str__(self) -> str:
        return f'Test {self.test_name} for tracking {self.model.name}.'

import sys
import logging
import abc
import warnings
from typing import TypeVar, Generic

import pandas as pd
from typing_extensions import override

import torch

from . import descriptors
from . import exceptions
from . import tracking
from . import io
from . import aggregator
from . import apply_ops
from . import protocols as p
from . import log_settings
from . import loading
from . import registering

_Input = TypeVar('_Input', bound=p.InputType)
_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)

logger = logging.getLogger('dry_torch')


class Evaluation(Generic[_Input, _Target, _Output], metaclass=abc.ABCMeta):
    max_stored_output: int = sys.maxsize
    partition: descriptors.Split
    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        tracker: contain the module and the optimizing strategy.
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
        outputs_list:
            An instance of TorchDictList that stores the last test evaluation.
        _store_outputs: if the flag is active store the module outputs in the
            outputs_list attribute. Default to False.

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
            mixed_precision: bool = False,
            name: str = '',
    ) -> None:
        self.model = model
        self.name = name
        self._loader = loading.TqdmLoader[tuple[_Input, _Target]](loader)
        self._calculator = metrics_calc
        device_is_cuda = self.model.device.type == 'cuda'
        self._mixed_precision = mixed_precision and device_is_cuda
        self._metrics = aggregator.TorchAggregator()
        self.outputs_list = list[_Output]()
        return

    @property
    def model_tracker(self) -> tracking.ModelTracker:
        return tracking.Experiment.current().tracker[self.model.name]

    @property
    def partition_log(self) -> pd.DataFrame:
        return self.model_tracker.log[self.partition]

    @partition_log.setter
    def partition_log(self, value) -> None:
        self.model_tracker.log[self.partition] = value
        return

    @property
    def metrics(self) -> dict[str, float]:
        out = self._metrics.reduce()
        return out

    @abc.abstractmethod
    def __call__(self, store_outputs: bool = False) -> None:
        ...

    def log_metrics(self) -> None:

        log_msg_list: list[str] = ['%(desc)-24s']
        desc = f'Average {self.partition.name.lower()} metrics:'
        log_args: dict[str, str | float] = {'desc': desc}
        self._update_partition_log()
        for metric, value in self.metrics.items():
            log_msg_list.append(f'%({metric})16s: %({metric}_value)4e')
            log_args.update({metric: metric, f'{metric}_value': value})
        logger.log(log_settings.INFO_LEVELS.metrics,
                   '\t'.join(log_msg_list),
                   log_args)
        self._metrics.clear()
        return

    def _run_epoch(self, store_outputs: bool):
        self.outputs_list.clear()
        for batch in self._loader:
            batch = apply_ops.apply_to(batch, self.model.device)
            self._run_batch(*batch, store_outputs=store_outputs)
            self._calculator.reset_calculated()
        self.log_metrics()

    def _run_batch(self,
                   inputs: _Input,
                   targets: _Target,
                   store_outputs: bool) -> None:
        with torch.autocast(device_type=self.model.device.type,
                            enabled=self._mixed_precision):
            outputs = self.model(inputs)
            self._calculator.calculate(outputs, targets)
            if store_outputs:
                self._store(outputs)
        self._metrics += self._calculator.metrics

    def _store(self, outputs: _Output) -> None:
        try:
            outputs = apply_ops.apply_cpu_detach(outputs)
        except (exceptions.FuncNotApplicableError,
                exceptions.NamedTupleOnlyError) as err:
            warnings.warn(exceptions.CannotStoreOutputWarning(str(err)))
        else:
            self.outputs_list.append(outputs)

    def _update_partition_log(self) -> None:
        info = {'Source': [self.name], 'Epoch': [self.model_tracker.epoch]}
        log_line = pd.DataFrame(
            info | {key: [value] for key, value in self.metrics.items()}
        )
        self.partition_log = pd.concat([self.partition_log, log_line])
        return

    def __str__(self) -> str:
        return f'Base Evaluator for {self.model.name}.'


class Diagnostic(Evaluation[_Input, _Target, _Output]):
    partition = descriptors.Split.TRAIN

    @override
    @torch.inference_mode()
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        self.model.module.eval()
        self._run_epoch(store_outputs)
        return

    @override
    def _update_partition_log(self) -> None:
        return


class Validation(Evaluation[_Input, _Target, _Output]):
    partition = descriptors.Split.VAL

    @registering.register_kwargs
    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            metrics_calc: p.MetricsCalculatorProtocol[_Output, _Target],
            mixed_precision: bool = False,
            name: str = '',
    ) -> None:
        super().__init__(model,
                         name=name,
                         loader=loader,
                         metrics_calc=metrics_calc,
                         mixed_precision=mixed_precision)
        return

    @override
    @torch.inference_mode()
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        self.model.module.eval()
        self._run_epoch(store_outputs)
        return

    def __str__(self) -> str:
        return f'Validator for {self.model.name}.'


class Test(Evaluation[_Input, _Target, _Output]):
    partition = descriptors.Split.TEST

    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        tracker: contain the module and the optimizing strategy.
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
        outputs_list:
            An instance of TorchDictList that stores the last test evaluation.
        _store_outputs: if the flag is active store the module outputs in the
            outputs_list attribute. Default to False.

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

    @registering.register_kwargs
    def __init__(
            self,
            model: p.ModelProtocol[_Input, _Output],
            /,
            *,
            loader: p.LoaderProtocol[tuple[_Input, _Target]],
            metrics_calc: p.MetricsCalculatorProtocol[_Output, _Target],
            name: str = '',
    ) -> None:
        super().__init__(model,
                         name=name,
                         loader=loader,
                         metrics_calc=metrics_calc)
        self._checkpoint = io.LogIO(model.name)
        return

    @override
    @torch.inference_mode()
    def __call__(self, store_outputs: bool = False) -> None:
        """
        Evaluates the module's performance on the specified partition of the
        dataset.

        Parameters:

        """
        logger.log(log_settings.INFO_LEVELS.experiment,
                   'Testing %(model_name)s.',
                   {'model_name': self.model.name})
        self.model.module.eval()
        self._run_epoch(store_outputs)
        self._checkpoint.save()
        return

    def __str__(self) -> str:
        return (
            f'{repr(self.model_tracker.default_names[self.__class__.__name__])}'
            'for tracker {self.model.name}.'
        )

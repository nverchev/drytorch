import functools
import sys
import warnings
from collections import defaultdict
from typing import Literal, Callable, Optional, Generic, TypeVar, Self, Iterable

import torch
from torch import autocast, Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .experiment_tracker import ExperimentTracker
from .model_optimizer import ModelOptimizer
from .dict_list import TorchDictList
from .context_managers import UsuallyFalse
from .recursive_ops import recursive_to
from .data_manager import IndexedDataLoader
from .protocols import MetricsProtocol, LossAndMetricsProtocol

TensorOutputs = TypeVar('TensorOutputs',
                        bound=Iterable[tuple[str, Tensor | list[Tensor]]] | dict[str, Tensor | list[Tensor]])

TensorInputs = TypeVar('TensorInputs', bound=Tensor | list[Tensor] | tuple[Tensor, ...])
TensorTargets = TypeVar('TensorTargets', bound=Tensor | list[Tensor] | tuple[Tensor, ...])
TensorData = tuple[TensorInputs, TensorTargets]
IndexData = tuple[TensorData, tuple[torch.LongTensor,]]
BatchData = tuple[int, IndexData]
MetricFunction = Callable[[TensorOutputs, TensorTargets], MetricsProtocol]
LossFunction = Callable[[TensorOutputs, TensorTargets], LossAndMetricsProtocol]


class Trainer(Generic[TensorInputs, TensorTargets, TensorOutputs]):
    """
    Implement the standard Pytorch training and evaluation loop.

    Args:
        model_handler: contain the model and the optimizing strategy.
        exp_tracker: track of the metrics and setting of the experiment.
        data_manager: contain the training dataset, and, optionally, the validation and test datasets.
        loss_fun: the _loss function, which needs to return batched values as in LossAndMetricsProtocol.
        metrics_fun: the test metrics function, returning TestMetricsProtocol. If None, _loss will be used instead.
        amp: whether to use mixed precision computing. Optional, default to False.

    Attributes:
        test_outputs (TorchDictList): An instance of TorchDictList that stores the last test evaluation.
        quiet (UsuallyFalse): A flag that controls the amount of output during training.

    Methods:
        train: run the training session, optionally quickly evaluate on the validation dataset.
        test: evaluate on the specified partition of the dataset.
        hook_before_training_epoch: property for adding a hook before running the training session.
        hook_after_training_epoch: property for adding a hook after running the training session.
    """

    max_stored_output: int = sys.maxsize  # maximum amount of stored evaluated test samples
    tqdm_update_frequency = 10

    def __init__(self,
                 model_handler: ModelOptimizer[TensorInputs, TensorOutputs],
                 exp_tracker: ExperimentTracker,
                 data_manager: IndexedDataLoader[TensorData],
                 loss_fun: LossFunction,
                 metrics_fun: Optional[MetricFunction] = None,
                 amp: bool = False,
                 ) -> None:

        self._model_optimizer: ModelOptimizer[TensorInputs, TensorOutputs] = model_handler
        self._exp_tracker: ExperimentTracker = exp_tracker
        self._data_manager = data_manager

        self._scaler = GradScaler(enabled=amp and model_handler.device.type == 'cuda')
        self._amp: bool = amp
        self._loss: LossFunction = loss_fun
        self._test_metrics: MetricFunction = metrics_fun if metrics_fun is not None else loss_fun

        self.quiet = UsuallyFalse()  # less output
        self._hook_before_training_epoch: Callable[[Self], None] = lambda instance: None
        self._hook_after_training_epoch: Callable[[Self], None] = lambda instance: None

        self.test_outputs: TorchDictList  # store last test evaluation
        return

    def train(self, num_epoch: int, val_after_train: bool = False) -> None:
        """
        Train the model for the specified number of epochs.

        Parameters:
            num_epoch: the number of epochs for which train the model.
            val_after_train: if the flag is active, evaluate loss function on the validation dataset. Default to False.
        """

        if not self.quiet:
            print('\rTraining {}'.format(self._model_optimizer))  # \r solves asynchronous stderr/stdout printouts
        for _ in range(num_epoch):
            self._model_optimizer.update_learning_rate()
            self._model_optimizer.epoch += 1
            print('\r====> Epoch:{:4d}'.format(self._model_optimizer.epoch), end=' ...  :' if self.quiet else '\n')
            self._model_optimizer.model.train()
            self.hook_before_training_epoch()
            self._run_epoch(partition='train')
            self.hook_after_training_epoch()
            if not self.quiet:
                print()
            if val_after_train:  # check losses on val
                self._model_optimizer.model.eval()
                with torch.inference_mode():
                    self._run_epoch(partition='val', use_test_metrics=False)
                if not self.quiet:
                    print()
        if self.quiet:
            print()
        return

    @torch.inference_mode()
    def test(self, partition: Literal['train', 'val', 'test'] = 'val', save_outputs: bool = False) -> None:
        """
        Evaluates the model's performance on the specified partition of the dataset.

        Parameters:
            partition: The partition of the dataset on which to evaluate the model's performance.  Default to 'val'.
            save_outputs: if the flag is active store the model outputs in the test_outputs attribute. Default to False.

        """
        self._model_optimizer.model.eval()
        self._run_epoch(partition=partition, save_outputs=save_outputs, use_test_metrics=True)
        print()
        return

    def _run_epoch(self, partition: Literal['train', 'val', 'test'],
                   save_outputs: bool = False,
                   use_test_metrics: bool = False) -> None:
        """
           Run a single epoch of training or evaluation.

           Parameters:
               partition: The partition of the dataset on which to evaluate the model's performance.
               save_outputs: if the flag is active, store the model outputs. Default to False.
               use_test_metrics: if the flag is active use the metrics function instead of the loss function.

           """
        try:
            loader: DataLoader[IndexData] = self._data_manager.loaders[partition]
        except KeyError:
            warnings.warn(f'Impossible to run model on the {partition} dataset: Dataset not found.')
            return
        log = self._exp_tracker.log[partition]
        try:
            num_batch: int = len(loader)
        except TypeError:
            num_batch = self._data_manager.max_num_batch
        if save_outputs:
            self.test_outputs = TorchDictList()

        epoch_log: defaultdict[str, float] = defaultdict(float)
        with tqdm(enumerate(loader), total=num_batch, disable=bool(self.quiet), file=sys.stderr) as tqdm_loader:
            epoch_seen = 0
            batch_data: BatchData
            for batch_data in tqdm_loader:
                (batch_idx, ((inputs, targets), (indices,))) = batch_data
                epoch_seen += indices.shape[0]
                batch_log, outputs, criterion = self._run_batch(inputs, targets, use_test_metrics)
                # if you get memory error, limit max_stored_output
                if save_outputs and self.max_stored_output >= epoch_seen:
                    self.test_outputs.extend(TorchDictList.from_batch(outputs, indices=(partition, indices)))
                for loss_or_metric, value in batch_log.items():
                    epoch_log[loss_or_metric] += value
                if batch_idx % (num_batch // self.tqdm_update_frequency or 1) == 0:  # or 1 prevents division by 0
                    tqdm_loader.set_postfix({'Seen': epoch_seen,
                                             'Loss': criterion})

        print(f'\rAverage {partition} {"metric(s)" if torch.is_inference_mode_enabled() else "loss_fun(es)"}:', end=' ')
        for loss_or_metric, value in epoch_log.items():
            value /= epoch_seen
            if torch.is_inference_mode_enabled() ^ (partition == 'train'):  # Evaluating train does not overwrite log
                log.loc[self._model_optimizer.epoch, loss_or_metric] = value
            print('{}: {:.4e}'.format(loss_or_metric, value), end='\t')

        return

    def _run_batch(self, inputs: TensorInputs, targets: TensorTargets, use_test_metrics: bool = False) \
            -> tuple[dict[str, float], TensorOutputs, float]:
        inputs, targets = recursive_to([inputs, targets], self._model_optimizer.device)
        with autocast(device_type=self._model_optimizer.device.type, enabled=self._amp):
            outputs: TensorOutputs = self._model_optimizer.model(inputs)
            if use_test_metrics:
                batched_performance = self._test_metrics(outputs, targets)
            else:
                batched_performance = self._loss(outputs, targets)
                criterion = batched_performance.criterion.mean(0)
        batch_log: dict[str, float] = {}
        for loss_or_metric, batched_value in batched_performance.metrics.items():
            batch_log[loss_or_metric] += batched_value.sum(0).item()
        if not torch.is_inference_mode_enabled():
            if torch.isnan(criterion):
                raise ValueError('Criterion is nan')
            if torch.isinf(criterion):
                raise ValueError('Criterion is inf')
            self._scaler.scale(criterion).backward()
            self._scaler.step(self._model_optimizer.optimizer)
            self._scaler.update()
            self._model_optimizer.optimizer.zero_grad()
        return batch_log, outputs, criterion.item()

    @property
    def hook_before_training_epoch(self) -> Callable[[], None]:
        """
        This hook is called before running the training session.
        """
        return functools.partial(self._hook_before_training_epoch, instance=self)

    @hook_before_training_epoch.setter
    def hook_before_training_epoch(self, value=Callable[[Self], None]) -> None:
        self._hook_before_training_epoch = value
        return

    @property
    def hook_after_training_epoch(self) -> Callable[[], None]:
        """
        This hook is called before running the training session.
        """
        return functools.partial(self._hook_after_training_epoch, instance=self)

    @hook_after_training_epoch.setter
    def hook_after_training_epoch(self, value=Callable[[Self], None]) -> None:
        self._hook_after_training_epoch = value
        return

    def __str__(self) -> str:
        return f'Trainer for {self._exp_tracker.exp_name}.'

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
from .model_handler import ModelHandler
from .dict_list import TorchDictList
from .context_managers import UsuallyFalse
from .recursive_ops import recursive_to
from .data_manager import DataManager
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
    Args:


        amp: mixed precision computing from torch.cuda.amp (works only with cuda)
                    scheduler: instance of a Scheduler class that modifies the learning rate depending on the epoch
        loss: this function may be a callable torch.nn.Module object (with reduction='none') or any function
                that returns a tensor of shape (batch_size, ) with the loss evaluation of individual samples.
                By default, its arguments are the model outputs and the targets. For more complex losses, and to
                compute metrics during training, read the documentation on the loss method and override it


    """
    max_stored_output: int = sys.maxsize  # maximum amount of stored evaluated test samples
    tqdm_update_frequency = 10

    def __init__(self,
                 model_handler: ModelHandler[TensorInputs, TensorOutputs],
                 exp_tracker: ExperimentTracker,
                 data_manager: DataManager[TensorData],
                 loss: LossFunction,
                 test_metrics: Optional[MetricFunction] = None,
                 amp: bool = False,
                 ) -> None:

        self.model_handler: ModelHandler[TensorInputs, TensorOutputs] = model_handler
        self.exp_tracker: ExperimentTracker = exp_tracker
        self.loaders = data_manager.loaders

        self.scaler = GradScaler(enabled=amp and model_handler.device.type == 'cuda')
        self.amp: bool = amp
        self.loss: LossFunction = loss
        self.test_metrics: MetricFunction = test_metrics if test_metrics is not None else loss

        self.test_indices: list[int] = []
        self.test_metadata: dict[str, str]
        self.test_outputs: TorchDictList  # store last test evaluation

        self.quiet = UsuallyFalse()  # less output
        self._hook_before_training_epoch: Callable[[Self], None] = lambda instance: None
        self._hook_after_training_epoch: Callable[[Self], None] = lambda instance: None
        return

    def train(self, num_epoch: int, val_after_train: bool = False) -> None:
        """
        It trains the model for the given epochs

        Args:
            num_epoch: a global learning rate for all the parameters or a list of dict with a 'lr' as a key
            val_after_train: run inference on the validation dataset after each epoch and calculates the loss
        """
        if not self.quiet:
            print('\rTraining {}'.format(self.model_handler))  # \r solves asynchronous stderr/stdout printouts
        for _ in range(num_epoch):
            self.model_handler.update_learning_rate()
            self.model_handler.epoch += 1
            print('\r====> Epoch:{:4d}'.format(self.model_handler.epoch), end=' ...  :' if self.quiet else '\n')
            self.model_handler.model.train()
            self.hook_before_training_epoch()
            self._run_epoch(partition='train')
            self.hook_after_training_epoch()
            if not self.quiet:
                print()
            if val_after_train:  # check losses on val
                self.model_handler.model.eval()
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
        It tests the current model in inference. It saves the metrics in saved_test_metrics

        Args:
            partition: the dataset used for testing
            save_outputs: if True, it stores the outputs of the model
        """
        self.model_handler.model.eval()
        self._run_epoch(partition=partition, save_outputs=save_outputs, use_test_metrics=True)
        print()
        return

    def _run_epoch(self, partition: Literal['train', 'val', 'test'],
                   save_outputs: bool = False,
                   use_test_metrics: bool = False) -> None:
        """
        It implements batching, backpropagation, printing and storage of the outputs

        Args:
            partition: the dataset used for the session
            save_outputs: if True, it stores the outputs of the model
            use_test_metrics: whether to use possibly computationally expensive test_metrics
        """

        loader: DataLoader[IndexData] = self.loaders[partition]
        log = self.exp_tracker.log[partition]

        num_batch: int = len(loader)
        if not num_batch:
            warnings.warn(f'Impossible to run model on the {partition} dataset: dataset not found.')
            return

        if save_outputs:
            self.test_indices = []
            self.test_outputs = TorchDictList()
            self.test_metadata = dict(partition=partition)

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
                    self.test_outputs.extend(TorchDictList.from_batch(outputs))
                    self.test_indices.extend(map(int, indices))
                for loss_or_metric, value in batch_log.items():
                    epoch_log[loss_or_metric] += value
                if batch_idx % (num_batch // self.tqdm_update_frequency or 1) == 0:  # or 1 prevents division by 0
                    tqdm_loader.set_postfix({'Seen': epoch_seen,
                                             'Loss': criterion})

        print(f'\rAverage {partition} {"metric(s)" if torch.is_inference_mode_enabled() else "loss(es)"}:', end=' ')
        for loss_or_metric, value in epoch_log.items():
            value /= epoch_seen
            if torch.is_inference_mode_enabled() ^ (partition == 'train'):  # Evaluating train does not overwrite log
                log.loc[self.model_handler.epoch, loss_or_metric] = value
            print('{}: {:.4e}'.format(loss_or_metric, value), end='\t')

        return

    def _run_batch(self, inputs: TensorInputs, targets: TensorTargets, use_test_metrics: bool = False) \
            -> tuple[dict[str, float], TensorOutputs, float]:
        inputs, targets = recursive_to([inputs, targets], self.model_handler.device)
        with autocast(device_type=self.model_handler.device.type, enabled=self.amp):
            outputs: TensorOutputs = self.model_handler.model(inputs)
            if use_test_metrics:
                batched_performance = self.test_metrics(outputs, targets)
            else:
                batched_performance = self.loss(outputs, targets)
                criterion = batched_performance.criterion.mean(0)
        batch_log: dict[str, float] = {}
        for loss_or_metric, batched_value in batched_performance.metrics.items():
            batch_log[loss_or_metric] += batched_value.sum(0).item()
        if not torch.is_inference_mode_enabled():
            if torch.isnan(criterion):
                raise ValueError('Criterion is nan')
            if torch.isinf(criterion):
                raise ValueError('Criterion is inf')
            self.scaler.scale(criterion).backward()
            self.scaler.step(self.model_handler.optimizer)
            self.scaler.update()
            self.model_handler.optimizer.zero_grad()
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
        return f'Trainer for {self.model_handler}.'

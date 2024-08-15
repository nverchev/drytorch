import logging
import sys
from typing import TypeVar, Iterator, Generator, Generic, Optional
from tqdm import auto

import torch
from torch.utils import data

from dry_torch import protocols as p
from dry_torch import exceptions
from dry_torch import log_settings

_Data_co = TypeVar('_Data_co',
                   bound=tuple[p.InputType, p.TargetType],
                   covariant=True)

logger = logging.getLogger('dry_torch')


class DataLoader(p.LoaderProtocol[_Data_co]):
    """
    A data-loader class with run_time settings
    .
    Args:
        dataset: dataset
        batch_size: the batch size.
    """

    def __init__(
            self,
            dataset: data.Dataset[_Data_co],
            batch_size: int,
    ) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_len = check_dataset_length(dataset)
        self._pin_memory: bool = torch.cuda.is_available()

    def get_loader(self) -> data.DataLoader[_Data_co]:
        inference = torch.is_inference_mode_enabled()
        drop_last: bool = not inference
        shuffle: bool = not inference
        loader = data.DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 drop_last=drop_last,
                                 shuffle=shuffle,
                                 pin_memory=self._pin_memory)
        return loader

    def __iter__(self) -> Iterator[_Data_co]:
        return self.get_loader().__iter__()

    def __len__(self) -> int:
        if torch.is_inference_mode_enabled():  # drop_last is true
            return self.dataset_len // self.batch_size
        return num_batches(self.dataset_len, self.batch_size)

    def set_pin_memory(self, value: bool) -> None:
        self._pin_memory = value


class TqdmLoader(Generic[_Data_co]):

    def __init__(
            self,
            loader: p.LoaderProtocol[_Data_co]) -> None:
        self.loader = loader
        self.batch_size = loader.batch_size or 0
        self.dataset_len = check_dataset_length(loader.dataset)
        self.disable_bar = logger.level > log_settings.INFO_LEVELS.tqdm_bar
        self._monitor_gen = _monitor()
        next(self._monitor_gen)
        self.seen_str = 'Seen'
        self.loss_str = 'Loss'

    def __iter__(self) -> Iterator[_Data_co]:
        num_batch = len(self.loader)
        with (auto.tqdm(enumerate(self.loader),
                        total=num_batch,
                        disable=self.disable_bar,
                        file=sys.stdout) as tqdm_loader):
            epoch_seen: int = 0
            batch_data: tuple[int, _Data_co]
            for batch_data in tqdm_loader:
                (batch_idx, batch) = batch_data
                yield batch
                epoch_seen += self.batch_size
                epoch_seen = min(epoch_seen, self.dataset_len)
                monitor_seen: dict[str, int] = {self.seen_str: epoch_seen}
                monitor_loss = next(self._monitor_gen)
                monitor_dict = monitor_seen | monitor_loss
                tqdm_loader.set_postfix(monitor_dict, refresh=False)

    def send(self, loss_value: float) -> None:
        self._monitor_gen.send({self.loss_str: loss_value})
        return

    def __len__(self) -> int:
        return len(self.loader)


def _monitor() -> Generator[dict[str, float], dict[str, float], None]:
    monitor_dict: Optional[dict[str, float]] = None
    while True:
        # if nothing is sent monitor_dict is None
        monitor_dict = yield monitor_dict or {}
        yield {}


def num_batches(dataset_len: int, batch_size: int) -> int:
    num_full_batches, last_batch_size = divmod(dataset_len, batch_size)
    return num_full_batches + bool(last_batch_size)


def check_dataset_length(dataset: data.Dataset) -> int:
    if hasattr(dataset, '__len__'):
        return dataset.__len__()
    raise exceptions.NoLengthError()

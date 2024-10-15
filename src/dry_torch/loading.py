import logging
import sys
from collections.abc import Iterator, Generator
from typing import TypeVar, Generic, Optional
from tqdm import auto

import torch
from torch.utils import data

from src.dry_torch import protocols as p
from src.dry_torch import exceptions

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


def num_batches(dataset_len: int, batch_size: int) -> int:
    num_full_batches, last_batch_size = divmod(dataset_len, batch_size)
    return num_full_batches + bool(last_batch_size)


def check_dataset_length(dataset: data.Dataset) -> int:
    if hasattr(dataset, '__len__'):
        return dataset.__len__()
    raise exceptions.DatasetHasNoLengthError()

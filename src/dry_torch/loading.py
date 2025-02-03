"""Classes for batching a dateset."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final, TypeVar, overload

import numpy as np
import torch
from torch.utils import data

from dry_torch import protocols as p
from dry_torch import exceptions

_Data_co = TypeVar('_Data_co',
                   bound=tuple[p.InputType, p.TargetType],
                   covariant=True)

_T = TypeVar('_T')


class Sliced(Sequence[_T]):
    """It slices a sequence keeping the reference to it."""

    def __init__(self,
                 seq: Sequence[_T],
                 slice_: slice) -> None:
        self.seq: Final = seq
        self.slice = slice_
        # take advantage of range implementation
        self.range = range(len(self.seq))[slice_]
        self.sliced = self.seq[slice_]

    def __len__(self) -> int:
        return len(self.range)

    @overload
    def __getitem__(self, idx: int) -> _T:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[_T]:
        ...

    def __getitem__(self, idx: int | slice) -> _T | Sequence[_T]:
        if isinstance(idx, int):
            return self.sliced[idx]
        else:  # slice
            # calculate new slice relative to original data
            new_range = self.range[idx]
            return Sliced(self.seq, self.range_to_slice(new_range))

    def __repr__(self) -> str:
        return self.seq.__repr__() + f'[{self.slice.__repr__()}]'

    @staticmethod
    def range_to_slice(r: range) -> slice:
        """Converts a range to the corresponding slice."""
        return slice(r.start, r.stop, r.step)


class Permutation(Sequence[int]):
    """Sliceable pseudo-random permutation."""

    def __init__(self,
                 size: int,
                 seed: int | None):
        self.size = size
        self.seed = np.random.randint(2 ** 16) if seed is None else seed
        rng = np.random.RandomState(self.seed)
        self._new_indices = rng.permutation(self.size).tolist()

    def __len__(self) -> int:
        return self.size

    @overload
    def __getitem__(self, idx: int) -> int:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[int]:
        ...

    def __getitem__(self, idx: int | slice) -> int | Sequence[int]:
        return self._new_indices[idx]

    def __repr__(self) -> str:
        return f"Permutation(size={self.size}, seed={self.seed})"


class DataLoader(p.LoaderProtocol[_Data_co]):
    """
    A data-loader class with runtime settings.

    This class wraps PyTorch's DataLoader with additional functionalities and
    annotations.

    Attributes:
        batch_size: Number of samples per batch.
        dataset: The dataset being loaded.
        dataset_len: Length of the dataset.
    """

    def __init__(
            self,
            dataset: data.Dataset[_Data_co],
            batch_size: int,
    ) -> None:
        """
        Args:
            dataset: The dataset to load data from.
            batch_size: Number of samples per batch.
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_len = check_dataset_length(dataset)
        self._pin_memory: bool = torch.cuda.is_available()

    def get_loader(self) -> data.DataLoader[_Data_co]:
        """
        Creates a DataLoader instance with runtime settings.

        Returns:
            A configured PyTorch DataLoader instance.
        """
        inference = torch.is_inference_mode_enabled()
        drop_last: bool = not inference
        shuffle: bool = not inference
        loader = data.DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 drop_last=drop_last,
                                 shuffle=shuffle,
                                 pin_memory=self._pin_memory)
        return loader

    def set_pin_memory(self, value: bool) -> None:
        """
        Sets whether to pin memory in GPU training.

        Args:
            value: If True, pin memory for faster GPU training.
        """
        self._pin_memory = value

    def split(
            self,
            split: float = 0.2,
            shuffle: bool = True,
            seed: int = 42,
    ) -> tuple[DataLoader[_Data_co], DataLoader[_Data_co]]:
        """
        Split loader into two.

        Args:
            split: fraction of the dataset to the second output loader.
            shuffle: whether to shuffle the data before splitting.
            seed: seed for shuffling.

        Returns:
            A tuple of (DataLoader, DataLoader).

        Raises:
            ValueError: If split is not between 0 and 1.
        """
        if split < 0 or split > 1:
            raise ValueError('split must be between 0 and 1.')

        dataset_size = check_dataset_length(self.dataset)
        second_size = int(dataset_size * split)
        first_size = dataset_size - second_size
        indices: Sequence[int]
        if shuffle:
            indices = Permutation(dataset_size, seed=seed)
        else:
            indices = range(dataset_size)

        first_dataset = data.Subset(
            self.dataset, Sliced(indices, slice(first_size))
        )
        second_dataset = data.Subset(
            self.dataset, Sliced(indices, slice(first_size, dataset_size))
        )

        first_loader = DataLoader(first_dataset, self.batch_size)
        second_loader = DataLoader(second_dataset, self.batch_size)

        return first_loader, second_loader

    def __iter__(self) -> Iterator[_Data_co]:
        return self.get_loader().__iter__()

    def __len__(self) -> int:
        if torch.is_inference_mode_enabled():  # drop_last is true
            return num_batches(self.dataset_len, self.batch_size)
        return self.dataset_len // self.batch_size


def num_batches(dataset_len: int, batch_size: int) -> int:
    """
    Calculates the number of batches in a dataset.

    Args:
        dataset_len: Length of the dataset.
        batch_size: Size of each batch.

    Returns:
        Total number of batches, including partial batches.
    """
    num_full_batches, last_batch_size = divmod(dataset_len, batch_size)
    return num_full_batches + bool(last_batch_size)


def check_dataset_length(dataset: data.Dataset) -> int:
    """
    Checks if a dataset has a valid length.

    Args:
        dataset: Dataset to check.

    Returns:
        Length of the dataset.

    Raises:
        DatasetHasNoLengthError: If the dataset has no __len__ method.
    """
    if hasattr(dataset, '__len__'):
        return dataset.__len__()
    raise exceptions.DatasetHasNoLengthError()

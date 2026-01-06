"""Module containing classes nad utilities for batching a dateset."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Final, TypeVar, overload

import numpy as np
import torch

from numpy import random
from torch import distributed as dist
from torch.utils import data
from typing_extensions import override

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.utils import apply_ops


__all__ = [
    'DataLoader',
    'take_from_dataset',
]

Data = TypeVar('Data', bound=tuple[p.InputType, p.TargetType], covariant=True)

_T = TypeVar('_T')

_default_device = torch.device('cpu')


class Sliced(Sequence[_T]):
    """Slice a sequence keeping the reference to it.

    Attributes:
        seq: the sequence to keep reference to.
        slice: the slice to use.
    """

    seq: Sequence[_T]
    slice: slice
    _range: range
    _sliced: Sequence[_T]

    def __init__(self, seq: Sequence[_T], slice_: slice) -> None:
        """Initialize.

        Args:
            seq: the sequence to keep reference to.
            slice_: the slice to use.
        """
        self.seq = seq
        self.slice = slice_
        # take advantage of range implementation
        self._range = range(len(self.seq))[slice_]
        self._sliced = self.seq[slice_]

    @overload
    def __getitem__(self, idx: int) -> _T: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[_T]: ...

    @override
    def __getitem__(self, idx: int | slice) -> _T | Sequence[_T]:
        if isinstance(idx, int):
            return self._sliced[idx]
        else:  # slice
            # calculate new slice relative to original data
            new_range = self._range[idx]
            return Sliced(self.seq, self.range_to_slice(new_range))

    @override
    def __len__(self) -> int:
        return len(self._range)

    @override
    def __repr__(self) -> str:
        return self.seq.__repr__() + f'[{self.slice.__repr__()}]'

    @staticmethod
    def range_to_slice(r: range) -> slice:
        """Convert a range to the corresponding slice."""
        return slice(r.start, r.stop, r.step)


class Permutation(Sequence[int]):
    """Sliceable pseudo-random permutation.

    Attributes:
        size: the length of the permutation.
        seed: seed for the random generator.
    """

    size: int
    seed: int
    _new_indices: list[int]

    def __init__(self, size: int, seed: int | None):
        """Initialize.

        Args:
            size: the length of the permutation.
            seed: seed for the random generator.
        """
        self.size = size
        self.seed = np.random.randint(2**16) if seed is None else seed
        rng = random.default_rng(self.seed)
        self._new_indices = rng.permutation(self.size).tolist()

    @overload
    def __getitem__(self, idx: int) -> int: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[int]: ...

    @override
    def __getitem__(self, idx: int | slice) -> int | Sequence[int]:
        return self._new_indices[idx]

    @override
    def __len__(self) -> int:
        return self.size

    @override
    def __repr__(self) -> str:
        return f'Permutation(size={self.size}, seed={self.seed})'


class DataLoader(p.LoaderProtocol[Data]):
    """A data-loader class with runtime settings.

    This class wraps PyTorch's DataLoader with additional functionalities.

    Attributes:
        batch_size: number of samples per batch.
        dataset: the dataset to load data from.
        dataset_len: length of the dataset.
        sampler: the sampling strategy for the dataset.
    """

    batch_size: int | None
    dataset: data.Dataset[Data]
    dataset_len: int
    sampler: data.Sampler | Iterable
    _pin_memory: bool
    _n_workers: int
    _distributed: bool
    _user_sampler: data.Sampler | Iterable | None

    def __init__(
        self,
        dataset: data.Dataset[Data],
        batch_size: int,
        pin_memory: bool | None = None,
        sampler: data.Sampler | Iterable[Any] | None = None,
        n_workers: int = 0,
    ) -> None:
        """Initialize.

        Args:
            dataset: the dataset to load data from.
            batch_size: number of samples per batch.
            pin_memory: pin memory for faster GPU training. Defaults to true
                when hardware acceleration is available.
            sampler: defines the strategy to draw samples from the dataset.
            n_workers: number of subprocesses for data loading.

        """
        self.batch_size: Final = batch_size
        self.dataset: Final = dataset
        self.dataset_len: Final = validate_dataset_length(dataset)
        acc_flag = torch.accelerator.is_available()
        self._pin_memory = acc_flag if pin_memory is None else pin_memory
        self._n_workers = n_workers
        self._distributed = dist.is_available() and dist.is_initialized()
        self._user_sampler = sampler
        self.sampler = self._init_sampler() if sampler is None else sampler
        return

    @override
    def __iter__(self) -> Iterator[Data]:
        inference = torch.is_inference_mode_enabled()
        return self.get_loader(inference).__iter__()

    @override
    def __len__(self) -> int:
        dataset_len = self.dataset_len
        batch_size = _validate_batch_size(self.batch_size)
        drop_last = torch.is_inference_mode_enabled()
        n_processes = dist.get_world_size() if self._distributed else 1
        return get_n_batches(dataset_len, batch_size, n_processes, drop_last)

    def get_dataset(self) -> data.Dataset[Data]:
        """Returns the dataset."""
        return self.dataset

    def get_loader(
        self,
        inference: bool,
    ) -> data.DataLoader[Data]:
        """Create a DataLoader instance with runtime settings.

        Args:
            inference: whether to use inference settings.
                Default checks torch global state.

        Returns:
            A configured PyTorch DataLoader instance.
        """
        if self._user_sampler is None:
            if inference:
                if isinstance(self.sampler, data.DistributedSampler):
                    sampler = data.DistributedSampler(
                        self.dataset, shuffle=False, drop_last=False
                    )
                else:
                    sampler = data.SequentialSampler(range(self.dataset_len))

            else:
                sampler = self.sampler

        else:
            sampler = self._user_sampler

        loader = data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            drop_last=not inference,
            sampler=sampler,
            pin_memory=self._pin_memory,
            num_workers=self._n_workers,
        )
        return loader

    def split(
        self,
        split: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[DataLoader[Data], DataLoader[Data]]:
        """Split the loader into two.

        Args:
            split: fraction of the dataset to the second output loader.
            shuffle: whether to shuffle the data before splitting.
            seed: seed for shuffling.

        Returns:
            A tuple of (DataLoader, DataLoader).

        Raises:
            ValueError: if split is not between 0 and 1.
        """
        if split < 0 or split > 1:
            raise ValueError('split must be between 0 and 1.')

        dataset_size = validate_dataset_length(self.dataset)
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
        batch_size = _validate_batch_size(self.batch_size)
        first_loader = DataLoader(first_dataset, batch_size)
        second_loader = DataLoader(second_dataset, batch_size)
        return first_loader, second_loader

    def _init_sampler(self) -> data.Sampler | Iterable:
        """Set the initial sampler for the loader."""
        if self._distributed:
            return data.DistributedSampler(self.dataset, drop_last=True)

        return data.RandomSampler(range(self.dataset_len))


def _validate_batch_size(batch_size: int | None) -> int:
    """Checks that the batch size is a valid number.

    Args:
        batch_size: the requested number of elements in the mini-batch.

    Returns:
        The verified size.

    Raises:
        InvalidBatchError: if the batch size is invalid.
    """
    if batch_size is None or batch_size < 1:
        raise ValueError('Batch size must be a positive integer. Got {}.')

    return batch_size


def validate_dataset_length(dataset: data.Dataset[Any]) -> int:
    """Checks if a dataset has a valid length.

    Args:
        dataset: dataset to check.

    Returns:
        Length of the dataset.

    Raises:
        DatasetHasNoLengthError: if the dataset has no __len__ method.
    """
    if get_length := getattr(dataset, '__len__', None):
        return get_length()

    raise exceptions.DatasetHasNoLengthError()


def get_n_batches(
    dataset_len: int, batch_size: int, n_processes: int, drop_last: bool
) -> int:
    """Calculate the number of batches in a dataset.

    Args:
        dataset_len: length of the dataset.
        batch_size: size of each batch.
        n_processes: number of processes used for data loading.
        drop_last: whether to drop the last batch if incomplete.

    Returns:
        Total number of batches, including partial batches.
    """
    n_batches, remainder = divmod(dataset_len, batch_size)
    if remainder and not drop_last:
        n_batches += 1

    n_rounds, remainder = divmod(n_batches, n_processes)
    if remainder and not drop_last:
        n_rounds += 1

    return n_rounds * n_processes


def take_from_dataset(
    dataset: data.Dataset[Data],
    n_samples: int = 1,
    preserve_order: bool = True,
    device: torch.device = _default_device,
) -> Data:
    """Sample a batch of elements from a dataset and transfers them to a device.

    Arguments:
        dataset: the dataset where to sample from.
        n_samples: the number of samples to take.
        preserve_order: take samples in order or randomly otherwise.
        device: device where to store the sample.

    Returns:
        The desired number of samples in a batch.
    """
    loader = data.DataLoader(
        dataset, batch_size=n_samples, shuffle=not preserve_order
    )
    return next(apply_ops.apply_to(batch, device) for batch in loader)

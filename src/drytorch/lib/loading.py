"""Classes and utilities for batching a dataset."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from typing import Any, Final, Generic, TypeVar, overload

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

    seq: Final[Sequence[_T]]
    slice: Final[slice]
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
        self._range = range(len(self.seq))[slice_]  # using range implementation
        self._sliced = self.seq[slice_]
        return

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
        seed: seed for the random generator.
        size: the length of the permutation.
    """

    seed: Final[int]
    size: Final[int]
    _new_indices: list[int]

    def __init__(self, size: int, seed: int | None) -> None:
        """Initialize.

        Args:
            size: the length of the permutation.
            seed: seed for the random generator.
        """
        if seed is None:
            seed = int(torch.randint(0, 1000000, (1,)).item())

        self.seed = seed
        self.size = size
        rng = random.default_rng(self.seed)
        self._new_indices = rng.permutation(self.size).tolist()
        return

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


class DataLoader(Generic[Data]):
    """A data-loader class with runtime settings.

    This class wraps PyTorch's DataLoader with additional functionalities.

    Attributes:
        batch_size: number of samples per batch.
        dataset: the dataset to load data from.
        dataset_len: length of the dataset.
        sampler: defines the strategy to draw samples from the dataset.
    """

    batch_size: Final[int | None]
    dataset: Final[data.Dataset[Data]]
    dataset_len: Final[int]
    sampler: Final[data.Sampler | Iterable]
    _distributed: bool
    _n_workers: int
    _pin_memory: bool
    _user_sampler: data.Sampler | Iterable | None

    def __init__(
        self,
        dataset: data.Dataset[Data],
        *,
        batch_size: int,
        n_workers: int = 0,
        pin_memory: bool | None = None,
        sampler: data.Sampler | Iterable[Any] | None = None,
    ) -> None:
        """Initialize.

        Args:
            dataset: the dataset to load data from.
            batch_size: number of samples per batch.
            n_workers: number of subprocesses for data loading.
            pin_memory: pin memory for faster GPU training. Defaults to true
                when hardware acceleration is available.
            sampler: defines the strategy to draw samples from the dataset.

        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_len = validate_dataset_length(dataset)
        acc_flag = torch.accelerator.is_available()
        self._pin_memory = acc_flag if pin_memory is None else pin_memory
        self._n_workers = n_workers
        self._distributed = dist.is_available() and dist.is_initialized()
        self._user_sampler = sampler
        self.sampler = self._init_sampler() if sampler is None else sampler
        return

    def __iter__(self) -> Iterator[Data]:
        """Return an iterator over the dataset in batches."""
        inference = torch.is_inference_mode_enabled()
        return self.get_loader(inference=inference).__iter__()

    def __len__(self) -> int:
        """Return the global number of batches in the dataset across all ranks.

        In distributed training, this is the total batch count across the entire
        world size, not just the number of batches yielded by the local process.
        """
        dataset_len = self.dataset_len
        batch_size = _validate_batch_size(self.batch_size)
        drop_last = not torch.is_inference_mode_enabled()
        n_processes = dist.get_world_size() if self._distributed else 1
        return get_n_batches(dataset_len, batch_size, n_processes, drop_last)

    def get_loader(
        self,
        *,
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
        ratio: float = 0.2,
        *,
        seed: int = 42,
        shuffle: bool = True,
    ) -> tuple[DataLoader[Data], DataLoader[Data]]:
        """Split the loader into two.

        Args:
            ratio: fraction of the dataset to the second output loader.
            seed: seed for shuffling.
            shuffle: whether to shuffle the data before splitting.

        Returns:
            A tuple of (DataLoader, DataLoader).

        Note:
            A user-provided sampler is not propagated: its indices refer to the
            original dataset, so each split loader builds its own.

        Raises:
            ValueError: if ratio is not between 0 and 1.
        """
        if ratio < 0 or ratio > 1:
            raise ValueError('ratio must be between 0 and 1.')

        dataset_size = validate_dataset_length(self.dataset)
        second_size = int(dataset_size * ratio)
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
        first_loader = DataLoader(
            first_dataset,
            batch_size=batch_size,
            n_workers=self._n_workers,
            pin_memory=self._pin_memory,
        )
        second_loader = DataLoader(
            second_dataset,
            batch_size=batch_size,
            n_workers=self._n_workers,
            pin_memory=self._pin_memory,
        )
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
        msg = 'Batch size must be a positive integer. Got {batch_size}.'
        raise ValueError(msg.format(batch_size=batch_size))

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
    if get_length := getattr(type(dataset), '__len__', None):
        return get_length(dataset)

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
    *,
    collate_fn: Callable[[list[Data]], Data] | None = None,
    device: torch.device = _default_device,
    indices: Sequence[int] | None = None,
    n_samples: int | None = None,
    preserve_order: bool = True,
) -> Data:
    """Take a batch of samples from a dataset and move it to a device.

    Args:
        dataset: the dataset where to take the samples from.
        collate_fn: function combining the samples into a batch. Defaults to
            the one of the PyTorch DataLoader.
        device: device where to store the batch.
        indices: the samples to take, in this order. Defaults to taking
            ``n_samples`` samples.
        n_samples: the number of samples to take when ``indices`` is not
            given. Defaults to one.
        preserve_order: when ``indices`` is not given, take the first samples,
            or random ones otherwise.

    Returns:
        The samples, collated into a batch.

    Raises:
        ValueError: if ``indices`` is given with ``n_samples`` or with
            ``preserve_order=False``.
        IndexError: if an index is outside the dataset.
    """
    if indices is not None:
        if n_samples is not None or not preserve_order:
            msg = 'indices set the samples and their order on their own.'
            raise ValueError(msg)

        if isinstance(dataset, Sized):
            size = len(dataset)
            for index in indices:
                if not 0 <= index < size:
                    msg = f'index {index} is outside the {size} samples.'
                    raise IndexError(msg)

        dataset = data.Subset(dataset, indices)
        n_samples = len(indices)

    batch_size = 1 if n_samples is None else n_samples
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not preserve_order,
        collate_fn=collate_fn,
    )
    return apply_ops.apply_to(next(iter(loader)), device)

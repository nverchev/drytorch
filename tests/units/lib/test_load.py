"""Tests for the "load" module."""

from collections.abc import Sequence

import torch

from torch.utils import data
from typing_extensions import override

import pytest

from drytorch.core import exceptions
from drytorch.lib.load import (
    DataLoader,
    Permutation,
    Sliced,
    num_batches,
    validate_dataset_length,
)


class SimpleDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple dataset for testing purposes."""

    def __init__(self, dataset: Sequence[tuple[int, int]]):
        """Constructor."""
        self.data = dataset

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.data[index]
        return torch.FloatTensor([out[0]]), torch.FloatTensor([out[1]])

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.data)


@pytest.fixture(scope='module')
def simple_seq() -> Sequence[tuple[int, int]]:
    """Provide a simple sequence for testing."""
    return [(i, i * 2) for i in range(10)]


class TestSliced:
    """Test Sliced class functionality."""

    @pytest.mark.parametrize(
        'slice_, expected',
        [
            (slice(0, 5), [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]),
            (slice(5, 10), [(5, 10), (6, 12), (7, 14), (8, 16), (9, 18)]),
            (slice(0, 0), []),
        ],
    )
    def test_sliced(self, simple_seq, slice_, expected) -> None:
        """Test slicing functionality of the Sliced class."""
        sliced = Sliced(simple_seq, slice_)
        assert list(sliced) == expected

    def test_sliced_chained(self) -> None:
        """Test chaining slices on the Sliced class."""
        seq = list(range(10))
        s1 = Sliced(seq, slice(2, 8))  # [2,3,4,5,6,7]
        s2 = s1[1:4]  # should be [3,4,5]
        assert len(s2) == 3
        assert s2[0] == 3
        assert s2[-1] == 5

    def test_sliced_chained_with_step(self) -> None:
        """Test chaining slices with a step on the Sliced class."""
        seq = list(range(10))
        s1 = Sliced(seq, slice(2, 8, 2))  # [2,4,6]
        s2 = s1[::2]  # should be [2,6]
        assert len(s2) == 2
        assert s2[0] == 2
        assert s2[-1] == 6


# Test class for Permutation
class TestPermutation:
    """Test Permutation class functionality."""

    def test_permutation(self) -> None:
        """Test Permutation class generates valid permutations."""
        perm = Permutation(10, seed=42)
        assert len(perm) == 10
        assert sorted(perm) == list(range(10))

    def test_permutation_seed(self) -> None:
        """Test Permutation class produces deterministic results."""
        perm1 = Permutation(10, seed=42)
        perm2 = Permutation(10, seed=42)
        assert list(perm1) == list(perm2)


# Test class for DataLoader
class TestDataLoader:
    """Test DataLoader class functionality."""

    @pytest.fixture(autouse=True)
    def dataset(self, simple_seq) -> data.Dataset:
        """Set up a simple dataset for testing."""
        return SimpleDataset(simple_seq)

    @pytest.fixture(autouse=True)
    def loader(self, dataset) -> DataLoader:
        """Provide a simple dataset for testing."""
        return DataLoader(dataset, batch_size=3)

    def test_dataloader_length(self, loader) -> None:
        """Test DataLoader correctly calculates the number of batches."""
        assert len(loader) == 3
        with torch.inference_mode():
            assert len(loader) == 4

    def test_dataloader_iteration(self, simple_seq, loader) -> None:
        """Test iteration over batches in the DataLoader."""
        with torch.inference_mode():
            batches = list(iter(loader))
            # the last batch has 1 item
            assert batches[-1][0] == simple_seq[-1][0]

    @pytest.mark.parametrize('shuffle', (True, False))
    def test_dataloader_split(self, loader, shuffle: bool) -> None:
        """Test splitting the DataLoader into training and validation sets."""
        train_loader, val_loader = loader.split(split=0.3, shuffle=shuffle)
        assert len(train_loader) == 7 // 3
        assert len(val_loader) == 3 // 3

    @pytest.mark.parametrize('value', (1.2, -0.2))
    def test_dataloader_split_invalid_ratio(self, loader, value) -> None:
        """Test DataLoader.split raises an error with invalid split ratios."""
        with pytest.raises(ValueError):
            loader.split(split=value)


def test_check_dataset_length_fail() -> None:
    """Test check_dataset_length raises an error when no len is defined."""
    dataset = torch.utils.data.Dataset[None]()
    with pytest.raises(exceptions.DatasetHasNoLengthError):
        validate_dataset_length(dataset)


def test_num_batches() -> None:
    """Test num_batches calculates batch count correctly."""
    assert num_batches(10, 3) == 4
    assert num_batches(10, 5) == 2
    assert num_batches(0, 3) == 0

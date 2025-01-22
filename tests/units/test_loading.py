"""Test for the loading module."""

import pytest

from typing import Sequence

import torch
from torch.utils import data
from typing_extensions import reveal_type

from src.dry_torch import loading


class SimpleDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple dataset for testing purposes."""

    def __init__(self, dataset: Sequence[tuple[int, int]]):
        self.data = dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.data[index]
        return torch.FloatTensor([out[0]]), torch.FloatTensor([out[1]])

    def __len__(self) -> int:
        return len(self.data)


@pytest.fixture
def simple_seq() -> Sequence[tuple[int, int]]:
    """Simple sequence for testing purposes."""
    return [(i, i * 2) for i in range(10)]


@pytest.fixture
def simple_dataset(
        simple_seq
) -> data.Dataset[tuple[torch.Tensor, torch.Tensor]]:
    """Simple sequence for testing purposes."""
    return SimpleDataset(simple_seq)


# Tests for Sliced
@pytest.mark.parametrize('slice_, expected', [
    (slice(0, 5), [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]),
    (slice(5, 10), [(5, 10), (6, 12), (7, 14), (8, 16), (9, 18)]),
    (slice(0, 0), []),
])
def test_sliced(slice_, expected, simple_seq) -> None:
    sliced = loading.Sliced(simple_seq, slice_)
    assert list(sliced) == expected
    assert len(sliced) == len(expected)


def test_sliced_chained() -> None:
    seq = list(range(10))
    s1 = loading.Sliced(seq, slice(2, 8))  # [2,3,4,5,6,7]
    s2 = s1[1:4]  # should be [3,4,5]
    assert len(s2) == 3
    assert s2[0] == 3
    assert s2[-1] == 5


def test_sliced_chained_with_step() -> None:
    seq = list(range(10))
    s1 = loading.Sliced(seq, slice(2, 8, 2))  # [2,4,6]
    s2 = s1[::2]  # should be [2,6]
    assert len(s2) == 2
    assert s2[0] == 2
    assert s2[-1] == 6


# Tests for Permutation
def test_permutation() -> None:
    perm = loading.Permutation(10, seed=42)
    assert len(perm) == 10
    assert sorted(perm) == list(range(10))


def test_permutation_seed() -> None:
    perm1 = loading.Permutation(10, seed=42)
    perm2 = loading.Permutation(10, seed=42)
    assert list(perm1) == list(perm2)


# Tests for DataLoader
def test_dataloader_creation(
        simple_dataset: data.Dataset[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    loader = loading.DataLoader(simple_dataset, batch_size=2)
    # should be DataLoader[tuple[torch.Tensor, torch.Tensor]]
    reveal_type(loader)
    assert len(loader) == 5  # 10 items / batch_size 2


def test_dataloader_iteration(simple_dataset) -> None:
    loader = loading.DataLoader(simple_dataset, batch_size=3)
    batches = list(iter(loader))
    assert len(batches) == 3  # last is skipped
    with torch.inference_mode():
        batches = list(iter(loader))
        assert len(batches) == 4  # last has 1 item
        assert batches[-1][0] == 9
        assert batches[-1][1] == 18


@pytest.mark.parametrize('shuffle', (True, False))
def test_dataloader_split(simple_dataset, shuffle: bool) -> None:
    loader = loading.DataLoader(simple_dataset, batch_size=2)
    train_loader, val_loader = loader.split(split=0.3, shuffle=False)
    with torch.inference_mode():
        assert len(train_loader) == 7 // 2
        assert len(val_loader) == 3 // 2

        train_data = list(iter(train_loader))
        val_data = list(iter(val_loader))

        assert len(train_data) == 4  # 7 items -> 4 batches
        assert len(val_data) == 2  # 3 items -> 2 batches


def test_dataloader_split_invalid_ratio(simple_dataset) -> None:
    loader = loading.DataLoader(simple_dataset, batch_size=2)

    with pytest.raises(ValueError):
        loader.split(split=1.5)


# Tests for utility functions
def test_num_batches() -> None:
    assert loading.num_batches(10, 3) == 4
    assert loading.num_batches(10, 5) == 2
    assert loading.num_batches(0, 3) == 0

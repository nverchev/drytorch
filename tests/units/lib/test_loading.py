"""Tests for the ``loading`` module."""

import itertools

from collections.abc import Sequence

import torch

from torch.utils import data
from typing_extensions import override

import pytest

from drytorch.core import exceptions
from drytorch.lib.loading import (
    DataLoader,
    Permutation,
    Sliced,
    get_n_batches,
    take_from_dataset,
    validate_dataset_length,
)


class SimpleDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple dataset for testing purposes."""

    def __init__(self, dataset: Sequence[tuple[int, int]]):
        """Initialize."""
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


def _concatenate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch, strict=True)
    return torch.cat(inputs), torch.cat(targets)


class TestSliced:
    """Test Sliced class functionality."""

    @pytest.mark.parametrize(
        ('slice_', 'expected'),
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


class TestDataLoader:
    """Test DataLoader class functionality."""

    @pytest.fixture
    def dataset(self, simple_seq) -> data.Dataset:
        """Set up a simple dataset for testing."""
        return SimpleDataset(simple_seq)

    @pytest.fixture
    def custom_sampler(self) -> data.Sampler:
        """Provide a custom sampler for testing."""
        return data.SequentialSampler(range(1))

    @pytest.fixture
    def loader(self, dataset) -> DataLoader:
        """Provide a DataLoader for testing."""
        return DataLoader(dataset, batch_size=3, n_workers=2)

    @pytest.fixture(autouse=True)
    def loader_with_sampler(self, dataset, custom_sampler) -> DataLoader:
        """Provide a DataLoader for testing with a custom sampler."""
        return DataLoader(dataset, batch_size=3, sampler=custom_sampler)

    def test_dataloader_sampler_initialized(self, loader) -> None:
        """Test that the sampler is always initialized."""
        assert loader.sampler is not None

    def test_dataloader_pin_memory_default(self, loader) -> None:
        """Test pin_memory defaults based on accelerator availability."""
        expected = torch.accelerator.is_available()

        assert loader._pin_memory == expected

    def test_dataloader_length_training(self, loader) -> None:
        """Test DataLoader batch count in training mode."""
        assert len(loader) == 3

    def test_dataloader_length_inference(self, loader) -> None:
        """Test DataLoader batch count in inference mode."""
        with torch.inference_mode():
            assert len(loader) == 4

    def test_dataloader_custom_sampler_preserved(
        self, loader_with_sampler, custom_sampler
    ) -> None:
        """Test that a custom sampler is preserved."""
        train_loader = loader_with_sampler.get_loader(inference=False)
        eval_loader = loader_with_sampler.get_loader(inference=True)

        assert train_loader.sampler is custom_sampler
        assert eval_loader.sampler is custom_sampler

    def test_sampler_switches_with_inference_mode(self, loader) -> None:
        """Test switch between Random and Sequential based on inference mode."""
        train_loader = loader.get_loader(inference=False)
        eval_loader = loader.get_loader(inference=True)

        assert isinstance(train_loader.sampler, data.RandomSampler)
        assert isinstance(eval_loader.sampler, data.SequentialSampler)

    def test_dataloader_iteration(self, simple_seq, loader) -> None:
        """Test iteration over batches in the DataLoader."""
        with torch.inference_mode():
            batches = list(iter(loader))

            assert batches[-1][0] == simple_seq[-1][0]

    def test_dataloader_n_workers(self, loader) -> None:
        """Test n_workers parameter is properly set."""
        assert loader.get_loader(inference=True).num_workers == 2

    def test_dataloader_drop_last_behavior(self, loader) -> None:
        """Test drop_last changes based on inference mode."""
        train_loader = loader.get_loader(inference=False)
        eval_loader = loader.get_loader(inference=True)

        assert train_loader.drop_last is True
        assert eval_loader.drop_last is False

    @pytest.mark.parametrize('shuffle', (True, False))
    def test_dataloader_split(self, loader, shuffle: bool) -> None:
        """Test splitting the DataLoader into training and validation sets."""
        train_loader, val_loader = loader.split(ratio=0.3, shuffle=shuffle)

        assert len(train_loader) == 7 // 3
        assert len(val_loader) == 3 // 3

    @pytest.mark.parametrize('value', (1.2, -0.2))
    def test_dataloader_split_invalid_ratio(self, loader, value) -> None:
        """Test DataLoader.split raises an error with invalid split ratios."""
        with pytest.raises(ValueError):
            loader.split(ratio=value)

    def test_dataloader_fold_validate_on_disjoint_parts(self, loader) -> None:
        """Validation parts cover the dataset once; training uses the rest."""
        folds = loader.fold(n_folds=3)

        val_parts = [list(val.dataset.indices) for _, val in folds]
        train_parts = [list(train.dataset.indices) for train, _ in folds]
        assert sorted(len(part) for part in val_parts) == [3, 3, 4]
        assert sorted(itertools.chain(*val_parts)) == list(range(10))
        for train_part, val_part in zip(train_parts, val_parts, strict=True):
            assert sorted(train_part + val_part) == list(range(10))

    def test_dataloader_fold_without_shuffling(self, loader) -> None:
        """Without shuffling, each fold validates on a contiguous part."""
        folds = loader.fold(n_folds=2, shuffle=False)

        assert list(folds[0][1].dataset.indices) == [0, 1, 2, 3, 4]
        assert list(folds[1][1].dataset.indices) == [5, 6, 7, 8, 9]

    @pytest.mark.parametrize('n_folds', (1, 11))
    def test_dataloader_fold_invalid_number(self, loader, n_folds) -> None:
        """Fewer than two folds, or more than the samples, are rejected."""
        with pytest.raises(ValueError, match='n_folds'):
            loader.fold(n_folds=n_folds)


def test_check_dataset_length_fail() -> None:
    """Test check_dataset_length raises an error when no len is defined."""
    dataset = torch.utils.data.Dataset[None]()

    with pytest.raises(exceptions.DatasetHasNoLengthError):
        validate_dataset_length(dataset)


@pytest.mark.parametrize(
    ('n_samples', 'batch_size', 'n_batches', 'drop_last', 'expected'),
    [
        (10, 3, 3, False, 6),
        (10, 3, 2, False, 4),
        (11, 5, 2, False, 4),
        (10, 5, 2, False, 2),
        (0, 3, 1, False, 0),
        (10, 3, 3, True, 3),
        (11, 5, 2, True, 2),
        (10, 3, 2, True, 2),
        (10, 5, 2, True, 2),
        (0, 3, 1, True, 0),
    ],
)
def test_get_n_batches(
    n_samples: int,
    batch_size: int,
    n_batches: int,
    drop_last: bool,
    expected: int,
) -> None:
    """Test n_batches calculates batch count correctly."""
    assert (
        get_n_batches(n_samples, batch_size, n_batches, drop_last) == expected
    )


def test_take_from_dataset_takes_the_first_samples(
    simple_seq: Sequence[tuple[int, int]],
) -> None:
    """Without indices, the first samples are taken as a batch."""
    dataset = SimpleDataset(simple_seq)

    inputs, targets = take_from_dataset(dataset, n_samples=3)

    torch.testing.assert_close(inputs, torch.tensor([[0.0], [1.0], [2.0]]))
    torch.testing.assert_close(targets, torch.tensor([[0.0], [2.0], [4.0]]))


def test_take_from_dataset_takes_the_given_indices_in_order(
    simple_seq: Sequence[tuple[int, int]],
) -> None:
    """The samples at the indices are taken in the order of the indices."""
    dataset = SimpleDataset(simple_seq)

    inputs, _ = take_from_dataset(dataset, indices=[7, 2, 5])

    torch.testing.assert_close(inputs, torch.tensor([[7.0], [2.0], [5.0]]))


@pytest.mark.parametrize(
    ('n_samples', 'preserve_order'), [(2, True), (None, False)]
)
def test_take_from_dataset_rejects_settings_contradicting_indices(
    simple_seq: Sequence[tuple[int, int]],
    n_samples: int | None,
    preserve_order: bool,
) -> None:
    """Indices set the samples and their order, so other choices are errors."""
    dataset = SimpleDataset(simple_seq)

    with pytest.raises(ValueError, match='indices'):
        take_from_dataset(
            dataset,
            indices=[1],
            n_samples=n_samples,
            preserve_order=preserve_order,
        )


def test_take_from_dataset_rejects_an_index_outside_the_dataset(
    simple_seq: Sequence[tuple[int, int]],
) -> None:
    """An index past the end of the dataset is reported with the length."""
    dataset = SimpleDataset(simple_seq)

    with pytest.raises(IndexError, match='10 samples'):
        take_from_dataset(dataset, indices=[1, 10])


def test_take_from_dataset_combines_the_samples_with_the_collate_fn(
    simple_seq: Sequence[tuple[int, int]],
) -> None:
    """A given collate function replaces the default stacking."""
    dataset = SimpleDataset(simple_seq)

    inputs, _ = take_from_dataset(
        dataset, indices=[4, 1], collate_fn=_concatenate
    )

    torch.testing.assert_close(inputs, torch.tensor([4.0, 1.0]))

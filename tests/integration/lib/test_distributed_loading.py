"""Integration tests for distributed data loading."""

import torch
import torch.distributed as dist

from ..conftest import DistributedWorker, RunningWorker
from torch import nn
from torch.utils import data
from typing_extensions import override

import pytest

from drytorch.lib.load import DataLoader
from drytorch.lib.models import Model
from drytorch.lib.runners import ModelRunner


class SimpleDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple dataset for testing purposes."""

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.FloatTensor([index])
        return out, out

    def __len__(self) -> int:
        """Number of samples."""
        return 16


@pytest.mark.skipif(not dist.is_available(), reason='Distributed not available')
class TestDistributedDataLoader:
    """Test distributed data loading functionality."""

    @pytest.fixture
    def world_size(self) -> int:
        """Number of processes to use for testing."""
        return 2

    @pytest.fixture
    def dataset(self) -> SimpleDataset:
        """Simple dataset for testing."""
        return SimpleDataset()

    def test_distributed_sampler_init(self, world_size, dataset) -> None:
        """Test that DistributedSampler is the default distributed setting."""
        worker = DistributedWorker(
            self._assert_sampler_init, world_size=world_size
        )
        exit_codes, _ = worker.process(dataset)
        assert all(exit_code == 0 for exit_code in exit_codes)

    def test_distributed_data_partitioning(self, world_size, dataset) -> None:
        """Test that data is partitioned correctly across processes."""
        worker = DistributedWorker(
            self._get_loader_indices, world_size=world_size
        )
        exit_codes, return_dict = worker.process(dataset)

        rank0_indices = set(return_dict[0])
        rank1_indices = set(return_dict[1])
        all_indices = rank0_indices | rank1_indices

        assert all(exit_code == 0 for exit_code in exit_codes)
        assert len(return_dict) == world_size
        assert len(all_indices) == len(dataset)  # true when (2 * batch) \ 16

    def test_distributed_sampler_shuffle_by_epoch(
        self, world_size, dataset
    ) -> None:
        """Test that different epochs produce different orderings."""
        worker = DistributedWorker(
            self._get_epoch_indices_by_epoch, world_size=world_size
        )
        exit_codes, return_dict = worker.process(dataset)

        assert all(exit_code == 0 for exit_code in exit_codes)

        # check that epochs produced different orderings
        for rank in range(world_size):
            epoch0_indices, epoch1_indices = return_dict[rank]
            assert epoch0_indices != epoch1_indices

    def test_model_runner_sets_distributed_epoch(
        self,
        world_size,
        dataset,
        tmp_path,
        example_run_id,
    ) -> None:
        """Test that ModelRunner correctly sets epoch on DistributedSampler."""
        running_worker = RunningWorker(
            self._assert_runner_set_epoch, tmp_path, example_run_id
        )
        worker = DistributedWorker(running_worker, world_size=world_size)
        exit_codes, _ = worker.process(dataset)

        assert all(exit_code == 0 for exit_code in exit_codes)

    @staticmethod
    def _assert_sampler_init(dataset: data.Dataset) -> None:
        """Assert that the loader is using a DistributedSampler."""
        loader = DataLoader(dataset, batch_size=4)
        assert isinstance(loader.sampler, data.DistributedSampler)
        assert loader.sampler.rank == dist.get_rank()

    @staticmethod
    def _get_loader_indices(dataset: data.Dataset) -> list[int]:
        """Get dataloader indices for the current process."""
        loader = DataLoader(dataset, batch_size=4)
        return [inpt.item() for (inputs, _) in loader for inpt in inputs]

    @staticmethod
    def _get_epoch_indices_by_epoch(
        dataset: data.Dataset,
    ) -> tuple[list[int], list[int]]:
        """Get tuple of indices in the first two epochs."""
        loader = DataLoader(dataset, batch_size=4)

        # epoch 0
        loader.sampler.set_epoch(0)
        epoch0_indices = [
            inpt.item() for (inputs, _) in loader for inpt in inputs
        ]

        # epoch 1 - should have different order due to different epoch
        loader.sampler.set_epoch(1)
        epoch1_indices = [
            inpt.item() for (inputs, _) in loader for inpt in inputs
        ]

        return epoch0_indices, epoch1_indices

    @staticmethod
    def _assert_runner_set_epoch(dataset: data.Dataset) -> None:
        """Test ModelRunner with distributed DataLoader."""
        loader = DataLoader(dataset, batch_size=4)
        model = Model(nn.Linear(1, 1), name='linear')
        model.epoch = 5
        runner = ModelRunner(model, loader=loader)
        runner()

        # Verify epoch was set
        assert loader.sampler.epoch == 5

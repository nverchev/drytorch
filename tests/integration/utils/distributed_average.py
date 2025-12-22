"""Tests for DistributedTorchAverager."""

import torch
import torch.distributed as dist

from ..conftest import DistributedWorker

import pytest

from drytorch.utils.average import TorchAverager


@pytest.mark.skipif(not dist.is_available(), reason='Distributed not available')
class TestDistributedTorchAverager:
    """Tests for DistributedTorchAverager."""

    @pytest.fixture
    def world_size(self) -> int:
        """Number of processes to use for testing."""
        return 2

    @pytest.fixture
    def averager(self) -> TorchAverager:
        """Fixture to create a DistributedTorchAverager instance."""
        return TorchAverager()

    def test_multiprocess_aggregate_and_count(self, world_size) -> None:
        """Test actual multiprocess synchronization."""
        worker = DistributedWorker(
            self._run_distributed_test, world_size=world_size
        )
        exit_codes, return_dict = worker.process()
        assert len(return_dict) == world_size
        assert all(exit_code == 0 for exit_code in exit_codes)
        for rank in range(world_size):
            assert return_dict[rank]['aggregate'] == 10.0  # 3 + 7 = 10
            assert return_dict[rank]['count'] == 4  # 2 + 2 = 4
            assert return_dict[rank]['average'] == 2.5  # 10 / 4 = 2.5

    def test_multiprocess_different_sizes(self, world_size) -> None:
        """Test with different tensor sizes per rank."""
        worker = DistributedWorker(
            self._run_test_different_sizes, world_size=world_size
        )
        exit_codes, return_dict = worker.process()

        assert all(exit_code == 0 for exit_code in exit_codes)
        assert len(return_dict) == world_size
        # the expected value is (3 + 12) / (2 + 3) = 3.0
        for rank in range(world_size):
            assert return_dict[rank]['average'] == 3.0

    @staticmethod
    def _run_distributed_test() -> dict[str, float | int]:
        """Helper function to run in each distributed process."""
        rank = dist.get_rank()
        averager = TorchAverager()
        tensor = torch.tensor([1.0 + rank * 2, 2.0 + rank * 2])
        averager._aggregate(tensor)
        averager._count(tensor)
        averager += {'my_metric': tensor}
        reduced = averager.all_reduce()
        return {
            'aggregate': averager.aggregate['my_metric'].item(),
            'count': averager.counts['my_metric'],
            'average': reduced['my_metric'].item(),
        }

    @staticmethod
    def _run_test_different_sizes() -> dict[str, float | int]:
        rank = dist.get_rank()
        averager = TorchAverager()
        if rank == 0:
            tensor = torch.tensor([1.0, 2.0])  # sum=3, count=2
        else:
            tensor = torch.tensor([3.0, 4.0, 5.0])  # sum=12, count=3

        averager += {'test': tensor}
        reduced = averager.all_reduce()
        return {'average': reduced['test'].item()}

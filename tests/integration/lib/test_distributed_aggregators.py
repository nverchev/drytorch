"""Tests for Aggregator with distributed processes."""

import torch
import torch.distributed as dist

from ..conftest import DistributedWorker

import pytest

from drytorch.lib.aggregators import Aggregator, TorchMeanAccumulator


@pytest.mark.skipif(not dist.is_available(), reason='Distributed not available')
class TestDistributedAggregator:
    """Tests for Aggregator with distributed processes."""

    @pytest.fixture
    def world_size(self) -> int:
        """Number of processes to use for testing."""
        return 2

    def test_multiprocess_aggregate_and_count(self, world_size) -> None:
        """Test actual multiprocess synchronization."""
        worker = DistributedWorker(
            self._run_distributed_test, world_size=world_size
        )
        exit_codes, return_dict = worker.process()
        assert len(return_dict) == world_size
        assert all(exit_code == 0 for exit_code in exit_codes)
        for rank in range(world_size):
            assert return_dict[rank]['total'] == 10.0  # 3 + 7 = 10
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
        aggregator = Aggregator[TorchMeanAccumulator]()
        tensor = torch.tensor([1.0 + rank * 2, 2.0 + rank * 2])
        aggregator += {'my_metric': TorchMeanAccumulator.from_value(tensor)}
        aggregator.sync()
        accumulator = aggregator.accumulators['my_metric']
        return {
            'total': accumulator.total.item(),
            'count': accumulator.count,
            'average': accumulator.reduce().item(),
        }

    @staticmethod
    def _run_test_different_sizes() -> dict[str, float | int]:
        rank = dist.get_rank()
        aggregator = Aggregator[TorchMeanAccumulator]()
        if rank == 0:
            tensor = torch.tensor([1.0, 2.0])  # sum=3, count=2
        else:
            tensor = torch.tensor([3.0, 4.0, 5.0])  # sum=12, count=3

        aggregator += {'test': TorchMeanAccumulator.from_value(tensor)}
        aggregator.sync()
        return {'average': aggregator.accumulators['test'].reduce().item()}

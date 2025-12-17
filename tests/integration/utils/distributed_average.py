"""Tests for DistributedTorchAverager."""

import os

from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pytest

from drytorch.utils.average import TorchAverager


if not torch.distributed.is_available():
    pytest.skip('PyTorch gloo backend not available', allow_module_level=True)


class TestDistributedTorchAverager:
    """Tests for DistributedTorchAverager."""

    @pytest.fixture
    def averager(self) -> TorchAverager:
        """Fixture to create a DistributedTorchAverager instance."""
        return TorchAverager()

    def test_multiprocess_aggregate_and_count(self) -> None:
        """Test actual multiprocess synchronization."""
        world_size = 2
        with mp.Manager() as manager:
            results = manager.dict()
            mp.spawn(
                self._run_distributed_test,
                args=(world_size, results),
                nprocs=world_size,
                join=True,
            )
            results = dict(results)
            assert len(results) == world_size
            for rank in range(world_size):
                assert results[rank]['aggregate'] == 10.0  # 3 + 7 = 10
                assert results[rank]['count'] == 4  # 2 + 2 = 4
                assert results[rank]['average'] == 2.5  # 10 / 4 = 2.5

    def test_multiprocess_different_sizes(self) -> None:
        """Test with different tensor sizes per rank."""
        world_size = 2
        with mp.Manager() as manager:
            results = manager.dict()
            mp.spawn(
                self._run_test_different_sizes,
                args=(world_size, results),
                nprocs=world_size,
                join=True,
            )
            results = dict(results)
            # the expected value is (3 + 12) / (2 + 3) = 3.0
            for rank in range(world_size):
                assert results[rank]['average'] == 3.0

    @staticmethod
    def _run_distributed_test(
        rank: int, world_size: int, results: dict[int, Any]
    ) -> None:
        """Helper function to run in each distributed process."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend='gloo',  # 'gloo' is for CPU
            rank=rank,
            world_size=world_size,
        )
        try:
            averager = TorchAverager()
            tensor = torch.tensor([1.0 + rank * 2, 2.0 + rank * 2])
            averager._aggregate(tensor)
            averager._count(tensor)
            averager += {'my_metric': tensor}
            reduced = averager.all_reduce()
            results[rank] = {
                'aggregate': averager.aggregate['my_metric'].item(),
                'count': averager.counts['my_metric'],
                'average': reduced['my_metric'].item(),
            }
        finally:
            dist.destroy_process_group()

    @staticmethod
    def _run_test_different_sizes(
        rank: int, world_size: int, results: dict[int, Any]
    ) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group(
            backend='gloo',
            rank=rank,
            world_size=world_size,
        )
        try:
            averager = TorchAverager()
            if rank == 0:
                tensor = torch.tensor([1.0, 2.0])  # sum=3, count=2
            else:
                tensor = torch.tensor([3.0, 4.0, 5.0])  # sum=12, count=3

            averager += {'test': tensor}
            reduced = averager.all_reduce()
            results[rank] = {'average': reduced['test'].item()}
        finally:
            dist.destroy_process_group()

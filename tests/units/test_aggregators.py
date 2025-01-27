"""Tests for the aggregators module."""

import pytest

import torch

from src.dry_torch import exceptions
from src.dry_torch.aggregators import Aggregator, Averager, TorchAverager


class TestAggregator:
    @pytest.fixture
    def aggregator(self) -> Aggregator[float]:
        """Fixture to create a base Aggregator instance (using Averager)."""
        return Averager()

    def test_add_single_value(self, aggregator: Aggregator[float]) -> None:
        """Test adding a single value to the aggregator."""
        aggregator['metric1'] = 2.0

        assert aggregator.aggregate['metric1'] == 2.0
        assert aggregator.counts['metric1'] == 1

    def test_add_multiple_values(self, aggregator: Aggregator[float]) -> None:
        """Test adding multiple values and checking aggregation."""
        aggregator['metric1'] = 2.0
        aggregator['metric2'] = 4.0

        assert aggregator.aggregate['metric1'] == 2.0
        assert aggregator.counts['metric1'] == 1
        assert aggregator.aggregate['metric2'] == 4.0
        assert aggregator.counts['metric2'] == 1

    def test_clear(self, aggregator: Aggregator[float]) -> None:
        """Test clearing the aggregator."""
        aggregator['metric1'] = 2.0
        aggregator.clear()

        assert not aggregator.aggregate
        assert not aggregator.counts
        assert not aggregator._cached_reduce

    def test_equality(self, aggregator: Aggregator[float]) -> None:
        """Test equality of two Aggregator instances with the same data."""
        other_aggregator = Averager()
        aggregator['metric1'] = 2.0
        other_aggregator['metric1'] = 2.0

        assert aggregator == other_aggregator

    def test_reduce(self, aggregator: Aggregator[float]) -> None:
        """Test reduce calculates averages for all metrics."""
        aggregator['metric1'] = 4.0
        aggregator['metric2'] = 6.0
        aggregator['metric3'] = 3.0

        expected_reduced = {
            'metric1': 4.0,
            'metric2': 6.0,
            'metric3': 3.0
        }

        assert aggregator.reduce() == expected_reduced
        assert aggregator._cached_reduce

    def test_cached_reduce(self, aggregator: Aggregator[float]) -> None:
        """Test cached_reduce stores result."""
        result = {'metric1': 4.0}
        aggregator._cached_reduce = result

        assert aggregator.reduce() == result


class TestAverager:
    @pytest.fixture
    def averager(self) -> Averager:
        """Fixture to create an Averager instance."""
        return Averager()

    def test_averager_single_value(self, averager: Averager) -> None:
        """Test adding a single float value to Averager."""
        averager['metric'] = 3.5
        assert averager.aggregate['metric'] == 3.5
        assert averager.counts['metric'] == 1

    def test_aggregate_function(self, averager: Averager) -> None:
        """Test _aggregate function returns the value itself for Averager."""
        assert averager._aggregate(5.0) == 5.0

    def test_count_function(self, averager: Averager) -> None:
        """Test _count function returns 1 for Averager."""
        assert averager._count(5.0) == 1


class TestTorchAverager:
    @pytest.fixture
    def torch_averager(self) -> TorchAverager:
        """Fixture to create a TorchAverager instance."""
        return TorchAverager()

    def test_invalid_tensor_shape(self, torch_averager: TorchAverager) -> None:
        """Test raise MetricsNotAVectorError for invalid tensor shape."""
        tensor = torch.ones((2, 2))  # Too many dimensions

        with pytest.raises(exceptions.MetricsNotAVectorError):
            torch_averager['metric'] = tensor

    def test_tensor_item(self, torch_averager: TorchAverager) -> None:
        """Test adding a 0-dimensional tensor to TorchAverager."""
        tensor = torch.tensor(1.0)
        torch_averager['metric'] = tensor

        assert torch_averager.aggregate['metric'] == tensor.sum().item()
        assert torch_averager.counts['metric'] == tensor.numel()

    def test_batched_tensor(self, torch_averager: TorchAverager) -> None:
        """Test TorchAverager handles batched tensors correctly."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch_averager['metric'] = tensor

        assert torch_averager.aggregate['metric'] == tensor.sum().item()
        assert torch_averager.counts['metric'] == tensor.numel()

    def test_reduce_with_tensors(self, torch_averager: TorchAverager) -> None:
        """Test reduce calculates averages for torch tensors."""
        tensor1 = torch.tensor([2.0, 3.0, 4.0])
        tensor2 = torch.tensor([1.0, 1.0, 1.0])

        torch_averager['metric1'] = tensor1
        torch_averager['metric2'] = tensor2

        expected_reduced = {
            'metric1': tensor1.sum().item() / tensor1.numel(),
            'metric2': tensor2.sum().item() / tensor2.numel(),
        }

        assert torch_averager.reduce() == expected_reduced

    def test_clear(self, torch_averager: TorchAverager) -> None:
        """Test that reduce returns an empty dict after clearing."""

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch_averager['metric'] = tensor
        torch_averager.clear()
        assert torch_averager.reduce() == {}

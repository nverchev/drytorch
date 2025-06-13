"""Tests for the "aggregators" module."""

import pytest

import torch

from drytorch.utils.aggregators import Aggregator
from drytorch.utils.aggregators import Averager
from drytorch.utils.aggregators import TorchAverager


class TestAggregator:
    @pytest.fixture
    def aggregator(self) -> Aggregator[float]:
        """Fixture to create a base Aggregator instance (using Averager)."""
        return Averager(metric1=2, metric2=4)

    def test_init(self, aggregator: Aggregator[float]) -> None:
        """Test adding multiple values and checking aggregation."""
        assert aggregator.aggregate['metric1'] == 2.0
        assert aggregator.counts['metric1'] == 1
        assert aggregator.aggregate['metric2'] == 4.0
        assert aggregator.counts['metric2'] == 1

    def test_clear(self, aggregator: Aggregator[float]) -> None:
        """Test clearing the aggregator."""
        aggregator.clear()

        assert not aggregator.aggregate
        assert not aggregator.counts
        assert not aggregator._cached_reduce

    def test_equality(self, aggregator: Aggregator[float]) -> None:
        """Test equality of two Aggregator instances with the same data."""
        other_aggregator = Averager(metric1=2, metric2=4)
        assert aggregator == other_aggregator
        different_aggregator = Averager(metric1=4, metric2=4)
        assert aggregator != different_aggregator

    def test_reduce(self, aggregator: Aggregator[float]) -> None:
        """Test reduce calculates averages for all metrics."""
        expected_reduced = {
            'metric1': 2.0,
            'metric2': 4.0,
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
        return Averager(metric1=2, metric2=4)

    def test_average(self, averager: Averager) -> None:
        """Test _aggregate function returns the value itself for Averager."""
        assert (averager + averager).reduce() == averager.reduce()


class TestTorchAverager:
    @pytest.fixture
    def torch_averager(self) -> TorchAverager:
        """Fixture to create a TorchAverager instance."""
        self.tensor = torch.ones((2, 2))
        return TorchAverager(metric=self.tensor)

    def test_init(self, torch_averager: TorchAverager) -> None:
        """Test TorchAverager handles batched tensors correctly."""
        assert torch_averager.aggregate['metric'] == self.tensor.sum().item()
        assert torch_averager.counts['metric'] == self.tensor.numel()

    def test_reduce_with_tensors(self, torch_averager: TorchAverager) -> None:
        """Test reduce calculates averages for torch tensors."""

        expected_reduced = {
            'metric': self.tensor.sum().item() / self.tensor.numel(),
        }

        assert torch_averager.reduce() == expected_reduced

    def test_clear(self, torch_averager: TorchAverager) -> None:
        """Test that reduce returns an empty dict after clearing."""
        torch_averager.clear()
        assert torch_averager.reduce() == {}

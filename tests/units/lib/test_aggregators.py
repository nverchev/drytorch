"""Tests for the "aggregators" module."""

import torch

import pytest

from drytorch.lib.aggregators import (
    Averager,
    TorchAverager,
)


class TestAggregator:
    """Tests for the AbstractAggregator using Averager."""

    @pytest.fixture
    def aggregator(self) -> Averager:
        """Fixture creating a base Averager instance."""
        return Averager(metric1=2.0, metric2=4.0)

    def test_init(self, aggregator: Averager) -> None:
        """Test initialization stores accumulators correctly."""
        assert aggregator.accumulators['metric1'].total == 2.0
        assert aggregator.accumulators['metric1'].count == 1
        assert aggregator.accumulators['metric2'].total == 4.0
        assert aggregator.accumulators['metric2'].count == 1

    def test_clear(self, aggregator: Averager) -> None:
        """Test clearing the aggregator."""
        aggregator.clear()
        assert not aggregator.accumulators
        assert not aggregator._cached_reduce

    def test_reduce(self, aggregator: Averager) -> None:
        """Test reduce calculates averages correctly."""
        expected = {'metric1': 2.0, 'metric2': 4.0}
        assert aggregator.reduce() == expected
        assert aggregator._cached_reduce

    def test_all_reduce(self, aggregator: Averager) -> None:
        """Test all_reduce recalculates after state change."""
        aggregator.reduce()
        aggregator.accumulators['metric1'].total = 4.0
        aggregator.accumulators['metric2'].total = 6.0
        expected = {'metric1': 4.0, 'metric2': 6.0}
        assert aggregator.all_reduce() == expected

    def test_cached_reduce(self, aggregator: Averager) -> None:
        """Test reduce returns cached result if present."""
        cached = {'metric1': 4.0}
        aggregator._cached_reduce = cached
        assert aggregator.reduce() == cached


class TestAverager:
    """Tests for Averager implementation."""

    @pytest.fixture
    def averager(self) -> Averager:
        """Fixture creating an Averager instance."""
        return Averager(metric1=2.0, metric2=4.0)

    def test_addition_preserves_mean(self, averager: Averager) -> None:
        """Test adding identical aggregators preserves mean."""
        combined = averager + averager
        assert combined.reduce() == averager.reduce()


class TestTorchAverager:
    """Tests for TorchAverager."""

    @pytest.fixture
    def torch_averager(self) -> TorchAverager:
        """Fixture creating a TorchAverager instance."""
        tensor = torch.ones((2, 2))
        return TorchAverager(metric=tensor)

    def test_init(self, torch_averager: TorchAverager) -> None:
        """Test tensor accumulator stores the correct total and count."""
        acc = torch_averager.accumulators['metric']
        assert torch.is_tensor(acc.total)
        assert acc.total.item() == 4.0
        assert acc.count == 4

    def test_reduce_with_tensors(self, torch_averager: TorchAverager) -> None:
        """Test reduce calculates correct tensor mean."""
        result = torch_averager.reduce()
        assert torch.is_tensor(result['metric'])
        assert result['metric'].item() == 1.0

    def test_clear(self, torch_averager: TorchAverager) -> None:
        """Test clearing removes all accumulators."""
        torch_averager.clear()
        assert torch_averager.reduce() == {}

    def test_all_reduce(self, torch_averager: TorchAverager) -> None:
        """Test all_reduce behaves like ``reduce`` when not distributed."""
        assert torch_averager.all_reduce() == torch_averager.reduce()

"""Tests for the "average" module."""

import math

import torch

import pytest

from drytorch.utils.average import (
    AbstractAverager,
    Averager,
    TorchAverager,
    get_moving_average,
    get_trailing_mean,
)


class TestAggregator:
    """Tests for the AbstractAverager using Averager."""

    @pytest.fixture
    def aggregator(self) -> AbstractAverager[float]:
        """Fixture to create a base Aggregator instance (using Averager)."""
        return Averager(metric1=2, metric2=4)

    def test_init(self, aggregator: AbstractAverager[float]) -> None:
        """Test adding multiple values and checking aggregation."""
        assert aggregator.aggregate['metric1'] == 2.0
        assert aggregator.counts['metric1'] == 1
        assert aggregator.aggregate['metric2'] == 4.0
        assert aggregator.counts['metric2'] == 1

    def test_clear(self, aggregator: AbstractAverager[float]) -> None:
        """Test clearing the aggregator."""
        aggregator.clear()
        assert not aggregator.aggregate
        assert not aggregator.counts
        assert not aggregator._cached_reduce

    def test_equality(self, aggregator: AbstractAverager[float]) -> None:
        """Test equality of two Aggregator instances with the same data."""
        other_aggregator = Averager(metric1=2, metric2=4)
        assert aggregator == other_aggregator
        different_aggregator = Averager(metric1=4, metric2=4)
        assert aggregator != different_aggregator

    def test_reduce(self, aggregator: AbstractAverager[float]) -> None:
        """Test reduce calculates averages for all metrics."""
        expected_reduced = {
            'metric1': 2.0,
            'metric2': 4.0,
        }
        assert aggregator.reduce() == expected_reduced
        assert aggregator._cached_reduce

    def test_cached_reduce(self, aggregator: AbstractAverager[float]) -> None:
        """Test cached_reduce stores result."""
        result = {'metric1': 4.0}
        aggregator._cached_reduce = result
        assert aggregator.reduce() == result


class TestAverager:
    """Tests for Averager implementation."""

    @pytest.fixture
    def averager(self) -> Averager:
        """Fixture to create an Averager instance."""
        return Averager(metric1=2, metric2=4)

    def test_average(self, averager: Averager) -> None:
        """Test _aggregate function returns the value itself for Averager."""
        assert (averager + averager).reduce() == averager.reduce()


class TestTorchAverager:
    """Tests for TorchAverager."""

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


def test_trailing_mean_full_window():
    """Test trailing mean with more elements than window size."""
    mean_fn = get_trailing_mean(3)
    assert mean_fn([1, 2, 3, 4, 5]) == pytest.approx((3 + 4 + 5) / 3)


def test_trailing_mean_exact_window():
    """Test trailing mean with sequence equal to window size."""
    mean_fn = get_trailing_mean(5)
    assert mean_fn([1, 2, 3, 4, 5]) == pytest.approx(3.0)


def test_trailing_mean_short_sequence():
    """Test trailing mean with fewer elements than window size."""
    mean_fn = get_trailing_mean(10)
    assert mean_fn([2, 4]) == pytest.approx(3.0)


def test_trailing_mean_window_one():
    """Test trailing mean with window size of 1."""
    mean_fn = get_trailing_mean(1)
    assert mean_fn([10, 20, 30]) == 30


def test_trailing_mean_empty_sequence():
    """Test trailing mean raises error on empty sequence."""
    mean_fn = get_trailing_mean(5)
    with pytest.raises(ZeroDivisionError):
        mean_fn([])


def test_trailing_mean_window_zero():
    """Test trailing mean raises error on zero window size."""
    with pytest.raises(ValueError):
        _ = get_trailing_mean(0)


def test_basic_average_equals_uniform():
    """Test moving average with uniform values."""
    ma = get_moving_average()
    assert ma([1, 1, 1, 1, 1]) == pytest.approx(1.0)


def test_weighted_average_behavior():
    """Test moving average weights recent values more."""
    ma = get_moving_average()
    result = ma([10, 1])
    assert 1 < result < 5.5


def test_threshold_effect():
    """Test mass_coverage effect on moving average."""
    seq = [1.0] * 10 + [10.0]
    ma_full = get_moving_average(decay=0.5, mass_coverage=1)
    ma_truncated = get_moving_average(decay=0.5, mass_coverage=0.999)
    full_result = ma_full(seq)
    truncated_result = ma_truncated(seq)
    assert truncated_result > full_result


@pytest.mark.parametrize(
    'decay, mass_coverage', [(0.9, 0.01), (0.8, 0.001), (0.99, 0.01)]
)
def test_threshold_formula(decay, mass_coverage):
    """Test that decay and coverage yield correct truncation cutoff."""
    cutoff_mass = 1 - mass_coverage
    cutoff_index = int(math.log(cutoff_mass, decay))
    tail_mass_before_cut = decay**cutoff_index
    tail_mass_after_cut = tail_mass_before_cut * decay
    assert tail_mass_before_cut >= cutoff_mass
    assert tail_mass_after_cut < cutoff_mass


def test_short_sequence():
    """Test moving average on single-element sequence."""
    ma = get_moving_average(decay=0.95, mass_coverage=0.999)
    result = ma([3.0])
    assert result == 3.0


@pytest.mark.parametrize(
    'decay, mass_coverage',
    [
        (0, 0.99),  # invalid decay (too low)
        (1, 0.99),  # invalid decay (too high)
        (0.9, 0.05),  # invalid mass_coverage (too low)
        (0.9, 1.1),  # invalid mass_coverage (too high)
    ],
)
def test_invalid_parameters(decay, mass_coverage):
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        get_moving_average(decay=decay, mass_coverage=mass_coverage)


def test_empty_input():
    """Test that moving average raises on empty input."""
    ma = get_moving_average()
    with pytest.raises(ZeroDivisionError):
        ma([])


def test_known_output():
    """Test moving average output against manually computed result."""
    ma = get_moving_average(decay=0.5, mass_coverage=1)
    weight_1 = 1
    weight_2 = 0.5
    expected = (4.0 * weight_1 + 2.0 * weight_2) / (weight_1 + weight_2)
    assert ma([2.0, 4.0]) == pytest.approx(expected)

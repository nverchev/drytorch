"""Tests for the "average" module."""

import math

import pytest

from drytorch.utils.averages import (
    get_moving_average,
    get_trailing_mean,
)


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

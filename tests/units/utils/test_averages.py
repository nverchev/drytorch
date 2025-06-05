"""Tests for the "averages" module."""

import pytest

from dry_torch.utils.averages import get_moving_average
from dry_torch.utils.averages import get_trailing_mean


def test_trailing_mean_full_window():
    mean_fn = get_trailing_mean(3)
    assert mean_fn([1, 2, 3, 4, 5]) == pytest.approx((3 + 4 + 5) / 3)


def test_trailing_mean_exact_window():
    mean_fn = get_trailing_mean(5)
    assert mean_fn([1, 2, 3, 4, 5]) == pytest.approx(3.0)


def test_trailing_mean_short_sequence():
    mean_fn = get_trailing_mean(10)
    assert mean_fn([2, 4]) == pytest.approx(3.0)  # average of all values


def test_trailing_mean_window_one():
    mean_fn = get_trailing_mean(1)
    assert mean_fn([10, 20, 30]) == 30  # only the last value


def test_trailing_mean_empty_sequence():
    mean_fn = get_trailing_mean(5)
    with pytest.raises(ZeroDivisionError):
        mean_fn([])


def test_trailing_mean_window_zero():
    with pytest.raises(ValueError):
        _ = get_trailing_mean(0)


def test_basic_average_equals_uniform():
    ma = get_moving_average()
    assert ma([1, 1, 1, 1, 1]) == pytest.approx(1.0)


def test_weighted_average_behavior():
    ma = get_moving_average()
    result = ma([10, 1])
    assert 1 < result < 5.5  # should be closer to 1 than 10


def test_threshold_effect():
    seq = [1.0] * 10 + [10.0]
    ma_full = get_moving_average(decay=0.5, threshold=0)
    ma_truncated = get_moving_average(decay=0.5, threshold=1e-2)

    full_result = ma_full(seq)
    truncated_result = ma_truncated(seq)

    # truncated version should give slightly more weight to the recent value
    assert truncated_result > full_result


def test_short_sequence():
    ma = get_moving_average(decay=0.95, threshold=1e-5)
    result = ma([3.0])
    assert result == 3.0


@pytest.mark.parametrize('decay, threshold', [
    (0, 1e-4),  # invalid decay (too low)
    (1, 1e-4),  # invalid decay (too high)
    (0.9, -0.1),  # negative threshold
    (0.9, 0.5),  # too large threshold for decay=0.9
])
def test_invalid_parameters(decay, threshold):
    with pytest.raises(ValueError):
        get_moving_average(decay=decay, threshold=threshold)


def test_empty_input():
    ma = get_moving_average()
    with pytest.raises(ZeroDivisionError):
        ma([])


def test_known_output():
    ma = get_moving_average(decay=0.5, threshold=0)
    weight_1 = 1
    weight_2 = 0.5
    expected = (4.0 * weight_1 + 2.0 * weight_2) / (weight_1 + weight_2)
    assert ma([2.0, 4.0]) == pytest.approx(expected)

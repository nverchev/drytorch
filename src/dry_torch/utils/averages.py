"""Averages for a sequence that can be used to monitor metrics."""

import math
from typing import Callable, Sequence


def get_moving_average(
        decay: float = 0.9,
        threshold: float = 1e-4,
) -> Callable[[Sequence[float]], float]:
    """
    Return a moving average by specifying the decay.

    Args:
        decay: the closer to 0 the more the last elements have weight.
        threshold: value below which the weight is considered negligible.

    Returns:
        The moving average function.

    Raises:
        ValueError if the decay is not between 0 and 1.
        ValueError if the threshold is negative or immediately hit.
    """
    if not 0 < decay < 1:
        raise ValueError('decay must be between 0 and 1.')

    if not 0 <= threshold <= decay * (1 - decay):
        raise ValueError('threshold should be small and non-negative.')

    # how far back to go back before the weight drops below the threshold
    if threshold:
        stop = -int(math.log(threshold / (1 - decay), decay)) - 2
    else:
        stop = None

    def _mean(float_list: Sequence[float], /) -> float:
        total: float = 0
        total_weights: float = 0  # should get close to one
        weight = 1 - decay  # weights are normalized
        for elem in float_list[:stop:-1]:
            total += weight * elem
            total_weights += weight
            weight *= decay

        return total / total_weights

    return _mean


def get_trailing_mean(window_size: int) -> Callable[[Sequence[float]], float]:
    """
    Return a trailing average by specifying window size.

    Args:
        window_size: number of items to aggregate.

    Returns:
        The windowed average function.

    Raises:
        ValueError if the window size is negative.
    """
    if window_size <= 0:
        raise ValueError('window_size must be positive.')

    def _mean(float_list: Sequence[float], /) -> float:
        clipped_window = min(window_size, len(float_list))
        return sum(float_list[-clipped_window:]) / clipped_window

    return _mean

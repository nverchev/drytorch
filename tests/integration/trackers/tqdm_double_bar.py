"""Tests training and epoch bars integration with the event system."""

import pytest

from typing import Generator

from dry_torch import tracking
from dry_torch import log_events
from dry_torch.trackers.tqdm import TqdmLogger
from dry_torch.trackers.tqdm import EpochBar
from dry_torch.trackers.tqdm import TrainingBar

expected = ('Epoch::   0%|\x1b[34m          \x1b[0m| 0/100\r'
            'Epoch: 5 / 100:   1%|\x1b[34m1         \x1b[0m| 1/100\n'
            '\r'
            '    test_source:   0%|\x1b[32m          \x1b[0m| 0/5\x1b[A\n'
            '\r'
            '                                     \x1b[A\r'
            'Epoch: 5 / 100:   2%|\x1b[34m2         \x1b[0m| 2/100\n'
            '\r'
            '    test_source:   0%|\x1b[32m          \x1b[0m| 0/5\x1b[A\n'
            '\r'
            '                                     \x1b[A\r'
            'Epoch: 5 / 100:   3%|\x1b[34m3         \x1b[0m| 3/100\n'
            '\r'
            '    test_source:   0%|\x1b[32m          \x1b[0m| 0/5\x1b[A\n'
            '\r'
            '                                     \x1b[A\r'
            'Epoch: 5 / 100:   3%|\x1b[34m3         \x1b[0m| 3/100')


@pytest.fixture(autouse=True)
def setup(monkeypatch) -> Generator[None, None, None]:
    """Set up a logger with temporary configuration."""
    # remove elapsed time prints for reproducibility
    epoch_bar_fmt = EpochBar.fmt
    EpochBar.fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}'
    training_bar_fmt = TrainingBar.fmt
    TrainingBar.fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}'
    yield

    EpochBar.fmt = epoch_bar_fmt
    TrainingBar.fmt = training_bar_fmt
    return


@pytest.fixture
def event_workflow(
        start_training_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        terminated_training_event,
        end_training_event,

) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    event_tuple = (
        start_training_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        terminated_training_event,
        end_training_event,
    )
    return event_tuple


def test_standard_trackers(event_workflow,
                           example_named_metrics,
                           string_stream):
    """Test the two bars are correctly updated by the events."""
    trackers = list[tracking.Tracker]()
    trackers.append(TqdmLogger(file=string_stream, enable_training_bar=True))
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    assert string_stream.getvalue().strip() == expected


def _notify_workflow(event_workflow: tuple[log_events.Event, ...],
                     trackers: list[tracking.Tracker],
                     example_named_metrics: dict[str, float]) -> None:
    for event in event_workflow:
        for tracker in trackers:
            tracker.notify(event)
            if isinstance(event, log_events.IterateBatch):
                for _ in range(event.num_iter):
                    event.update(example_named_metrics)
                event.push_updates.clear()  # necessary to reinitialize

    return

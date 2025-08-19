"""Tests TqdmLogger integration with the event system."""

from collections.abc import Generator, Sequence

import pytest

from drytorch.core import log_events, track


try:
    from drytorch.trackers.tqdm import EpochBar, TqdmLogger, TrainingBar
except ImportError:
    pytest.skip('tqdm not available', allow_module_level=True)
    raise


expected_out = (
    'Epoch::   0%|\x1b[34m          \x1b[0m| 0/3\r'
    'Epoch: 5 / 8:  33%|\x1b[34m###3      \x1b[0m| 1/3\n'
    '\r'
    '    test_source:   0%|\x1b[32m          \x1b[0m| 0/5\x1b[A\n'
    '\r'
    '                                     \x1b[A\r'
    'Epoch: 5 / 8:  67%|\x1b[34m######6   \x1b[0m| 2/3\n'
    '\r'
    '    test_source:   0%|\x1b[32m          \x1b[0m| 0/5\x1b[A\n'
    '\r'
    '                                     \x1b[A\r'
    'Epoch: 5 / 8: 100%|\x1b[34m##########\x1b[0m| 3/3\n'
    '\r'
    '    test_source:   0%|\x1b[32m          \x1b[0m| 0/5\x1b[A\n'
    '\r'
    '                                     \x1b[A\r'
    'Epoch: 5 / 8: 100%|\x1b[34m##########\x1b[0m| 3/3'
)


@pytest.fixture
def event_workflow(
    start_training_event,
    start_epoch_event,
    iterate_batch_event,
    metrics_event,
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
        metrics_event,
        end_epoch_event,
        start_epoch_event,
        iterate_batch_event,
        metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        start_epoch_event,
        iterate_batch_event,
        metrics_event,
        terminated_training_event,
        end_training_event,
    )
    return event_tuple


class TestTqdmLoggerFullCycle:
    """Complete TqdmLogger session and tests it afterward."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch) -> Generator[None, None, None]:
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

    def test_tqdm_logger_with_training_bar_output(
        self, event_workflow, example_named_metrics, string_stream
    ):
        """Test TqdmLogger with double bar produces the expected output."""
        trackers = [TqdmLogger(file=string_stream, enable_training_bar=True)]

        _notify_workflow(event_workflow, trackers, example_named_metrics)

        actual_output = string_stream.getvalue().strip()
        assert actual_output == expected_out


def _notify_workflow(
    event_workflow: tuple[log_events.Event, ...],
    trackers: Sequence[track.Tracker],
    example_named_metrics: dict[str, float],
) -> None:
    for event in event_workflow:
        for tracker in trackers:
            tracker.notify(event)
            if isinstance(event, log_events.IterateBatchEvent):
                for _ in range(event.num_iter):
                    event.update(example_named_metrics)
                event.push_updates.clear()  # necessary to reinitialize

    return

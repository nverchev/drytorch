"""Tests TqdmLogger integration with the event system."""

import io

from collections.abc import Generator, Sequence

import pytest

from drytorch.core import log_events, tracking


try:
    from drytorch.trackers.tqdm import EpochBar, TqdmLogger, TrainingBar
except ImportError:
    pytest.skip('tqdm not available', allow_module_level=True)
    raise


EXPECTED_OUT = (
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
        assert actual_output == EXPECTED_OUT


class TestTqdmLoggerPauseContinue:
    """Tests TqdmLogger bar lifecycle and isolation across interleaved runs."""

    @pytest.fixture
    def tracker(self, string_stream: io.StringIO) -> TqdmLogger:
        """Create a TqdmLogger writing to a StringIO stream."""
        return TqdmLogger(enable_training_bar=True, file=string_stream)

    def test_interleaved_pause_continue(
        self,
        tracker: TqdmLogger,
        start_training_event: log_events.StartTrainingEvent,
        iterate_batch_event: log_events.IterateBatchEvent,
        pause_experiment_event: log_events.PauseExperimentEvent,
        continue_experiment_event: log_events.ContinueExperimentEvent,
        stop_experiment_event: log_events.StopExperimentEvent,
        stop_experiment_event_b: log_events.StopExperimentEvent,
    ) -> None:
        """Verify TqdmLogger closes bars on pause and isolates runs."""
        # Trigger
        tracker.notify(start_training_event)
        tracker.notify(iterate_batch_event)
        bar_a_training = tracker._training_bar
        bar_a_epoch = tracker._epoch_bar

        tracker.notify(pause_experiment_event)
        training_bar_after_pause = tracker._training_bar
        epoch_bar_after_pause = tracker._epoch_bar

        tracker.notify(start_training_event)
        tracker.notify(iterate_batch_event)
        bar_b_training = tracker._training_bar
        bar_b_epoch = tracker._epoch_bar

        tracker.notify(stop_experiment_event_b)

        tracker.notify(continue_experiment_event)
        tracker.notify(iterate_batch_event)
        bar_a_continued_epoch = tracker._epoch_bar

        tracker.notify(stop_experiment_event)
        final_training_bar = tracker._training_bar
        final_epoch_bar = tracker._epoch_bar

        # Assert
        assert bar_a_training is not None
        assert bar_a_epoch is not None
        assert training_bar_after_pause is None
        assert epoch_bar_after_pause is None
        assert bar_b_training is not bar_a_training
        assert bar_b_epoch is not bar_a_epoch
        assert bar_a_continued_epoch is not bar_b_epoch
        assert final_training_bar is None
        assert final_epoch_bar is None


def _notify_workflow(
    event_workflow: tuple[log_events.Event, ...],
    trackers: Sequence[tracking.Tracker],
    example_named_metrics: dict[str, float],
) -> None:
    for event in event_workflow:
        for tracker in trackers:
            tracker.notify(event)
            if isinstance(event, log_events.IterateBatchEvent):
                for _ in range(event.n_iter):
                    event.update(example_named_metrics)
                event.push_updates.clear()  # necessary to reinitialize

    return

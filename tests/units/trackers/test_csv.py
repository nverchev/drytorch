"""Tests for the "csv" module."""

import copy

from collections.abc import Generator

import pytest

from drytorch.core import exceptions
from drytorch.core.exceptions import TrackerError
from drytorch.trackers.csv import CSVDumper


class TestCsvDumper:
    """Tests for the HydraLink tracker with actual Hydra integration."""

    @pytest.fixture
    def tracker(self, tmp_path) -> CSVDumper:
        """Set up the instance."""
        return CSVDumper(tmp_path)

    @pytest.fixture
    def tracker_started(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[CSVDumper, None, None]:
        """Start the instance."""
        tracker.notify(start_experiment_mock_event)
        yield tracker
        tracker.notify(stop_experiment_mock_event)
        return

    @pytest.fixture
    def tracker_started_with_resume(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[CSVDumper, None, None]:
        """Start the instance."""
        start_experiment_mock_event.resumed = True
        tracker.notify(start_experiment_mock_event)
        yield tracker
        tracker.notify(stop_experiment_mock_event)
        return

    def test_notify_metrics_event(
        self, tracker_started, epoch_metrics_mock_event
    ) -> None:
        """Test file is created."""
        tracker_started.notify(epoch_metrics_mock_event)
        csv_path = tracker_started._file_path(
            tracker_started._get_run_dir(),
            epoch_metrics_mock_event.model_name,
            epoch_metrics_mock_event.source_name,
        )
        assert csv_path.exists()

    def test_resume_with_header_mismatch(
        self,
        tracker_started_with_resume,
        epoch_metrics_mock_event,
        example_named_metrics,
    ) -> None:
        """Test a TrackerError is raised if headers do not match on resume."""
        run_dir = tracker_started_with_resume._get_run_dir()
        csv_path = tracker_started_with_resume._file_path(
            run_dir,
            epoch_metrics_mock_event.model_name,
            epoch_metrics_mock_event.source_name,
        )

        with csv_path.open('w') as f:
            f.write('"Model","Source","Epoch","DifferentMetric"\n')
            f.write('"model","source",1,0.1\n')

        with pytest.raises(TrackerError, match='headers'):
            tracker_started_with_resume.notify(epoch_metrics_mock_event)

    def test_read_csv(
        self, tracker_started, epoch_metrics_mock_event, example_named_metrics
    ) -> None:
        """Test read_csv gets the correct epochs."""
        for epoch in (1, 2, 3, 1, 2, 3):
            epoch_metrics_mock_event.epoch = epoch
            tracker_started.notify(epoch_metrics_mock_event)
            epochs, metric_dict = tracker_started.read_csv(
                epoch_metrics_mock_event.model_name,
                epoch_metrics_mock_event.source_name,
                max_epoch=2,
            )
        assert epochs == [1, 2]
        for metric, value in metric_dict.items():
            assert example_named_metrics[metric] == value[0] == value[1]

    def test_load_metrics(
        self, tracker, tracker_started_with_resume, epoch_metrics_mock_event
    ) -> None:
        """Test _load_metrics gets the correct epochs."""
        model_name = epoch_metrics_mock_event.model_name
        source_name = epoch_metrics_mock_event.source_name

        tracker.notify(epoch_metrics_mock_event)

        assert source_name in tracker._load_metrics(model_name)
        assert source_name in tracker_started_with_resume._load_metrics(
            model_name
        )
        with pytest.raises(TrackerError):
            _ = tracker_started_with_resume._load_metrics('wrong_name')

    def test_pause_stashes_and_continue_restores(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that pause stashes state and continue restores it."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)
        tracker.notify(continue_experiment_mock_event)

    def test_pause_start_stop_continue_keeps_state(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        stop_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that concurrent runs do not corrupt stashed state."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.run_id = 'run2'

        tracker.notify(start_2)

        stop_2 = copy.copy(stop_experiment_mock_event)
        stop_2.run_id = 'run2'
        tracker.notify(stop_2)

        tracker.notify(continue_experiment_mock_event)

    def test_continue_without_stash_raises_error(
        self, tracker, continue_experiment_mock_event
    ) -> None:
        """Test that continuing without a stash raises an error."""
        with pytest.raises(exceptions.NoStashedStateError):
            tracker.notify(continue_experiment_mock_event)

    def test_close_releases_stashes(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
    ) -> None:
        """Test that close unconditionally releases stashed resources."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)
        tracker.close()

"""Tests for the "csv" module."""

from collections.abc import Generator

import pytest

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

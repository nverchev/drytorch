"""Test "csv" module"""

import pytest

from drytorch.exceptions import TrackerException
from drytorch.trackers.csv import CSVDumper


class TestCsvDumper:
    """Tests for the HydraLink tracker with actual Hydra integration."""

    @pytest.fixture
    def tracker(self, tmp_path) -> CSVDumper:
        """Set up the instance."""
        return CSVDumper(tmp_path)

    @pytest.fixture
    def tracker_with_resume(self, tracker) -> CSVDumper:
        """Set up the instance with resume."""
        return CSVDumper(tracker.par_dir, resume_run=True)

    def test_file_name(self, tracker) -> None:
        """Test file name corresponds to expected."""
        file_address = tracker.file_name('model_name',
                                         'source_name')
        file_string = '/model_name/csv_metrics/source_name.csv'
        assert file_address.as_posix().endswith(file_string)

    def test_notify_metrics_event(self,
                                  tracker,
                                  epoch_metrics_mock_event) -> None:
        """Test file is created."""
        tracker.notify(epoch_metrics_mock_event)
        csv_path = tracker.file_name(epoch_metrics_mock_event.model_name,
                                     epoch_metrics_mock_event.source_name)
        assert csv_path.exists()

    def test_read_csv(self,
                      tracker,
                      epoch_metrics_mock_event,
                      example_named_metrics) -> None:
        """Test read_csv gets the correct epochs."""
        for epoch in (1, 2, 3, 1, 2, 3):
            epoch_metrics_mock_event.epoch = epoch
            tracker.notify(epoch_metrics_mock_event)
        epochs, metric_dict = tracker.read_csv(
            epoch_metrics_mock_event.model_name,
            epoch_metrics_mock_event.source_name,
            max_epoch=2,
        )
        assert epochs == [1, 2]
        for metric, value in metric_dict.items():
            assert example_named_metrics[metric] == value[0] == value[1]

    def test_load_metrics(self,
                          tracker,
                          tracker_with_resume,
                          epoch_metrics_mock_event) -> None:
        """Test _load_metrics gets the correct epochs."""
        model_name = epoch_metrics_mock_event.model_name
        source_name = epoch_metrics_mock_event.source_name
        assert tracker._load_metrics(model_name) == {}
        tracker.notify(epoch_metrics_mock_event)
        assert source_name in tracker._load_metrics(model_name)
        assert source_name in tracker_with_resume._load_metrics(model_name)
        with pytest.raises(TrackerException):
            _ = tracker_with_resume._load_metrics('wrong_name')

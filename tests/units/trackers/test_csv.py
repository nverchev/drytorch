"""Test csv dumper"""

import pytest

from src.dry_torch.trackers.csv import CSVDumper


class TestCsvDumper:
    """Tests for the HydraLink tracker with actual Hydra integration."""

    @pytest.fixture
    def tracker(self, start_experiment_mock_event, tmp_path) -> CSVDumper:
        """Set up the instance."""
        return CSVDumper(tmp_path)

    def test_notify_with_epoch_metrics_new_file(self,
                                                tracker,
                                                epoch_metrics_mock_event,
                                                example_named_metrics):
        tracker.notify(epoch_metrics_mock_event)
        csv_path = tracker.file_name(epoch_metrics_mock_event.model_name,
                                          epoch_metrics_mock_event.source_name)
        assert csv_path.exists()

        epochs, metric_dict = tracker.read_csv(
            epoch_metrics_mock_event.model_name,
            epoch_metrics_mock_event.source_name,
        )

        assert epochs[0] == epoch_metrics_mock_event.epoch
        for metric, value in metric_dict.items():
            assert example_named_metrics[metric] == value[0]

        epoch_metrics_mock_event.epoch += 1
        tracker.notify(epoch_metrics_mock_event)

        # Check the file was created with correct content
        csv_path = tracker.file_name(epoch_metrics_mock_event.model_name,
                                          epoch_metrics_mock_event.source_name)
        assert csv_path.exists()

        epochs, metric_dict = tracker.read_csv(
            epoch_metrics_mock_event.model_name,
            epoch_metrics_mock_event.source_name,
        )

        assert epochs[0] == epoch_metrics_mock_event.epoch - 1  # previous event
        for metric, value in metric_dict.items():
            assert example_named_metrics[metric] == value[0]

        assert epochs[1] == epoch_metrics_mock_event.epoch
        for metric, value in metric_dict.items():
            assert example_named_metrics[metric] == value[1]

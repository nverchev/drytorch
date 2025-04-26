"""Test csv dumper"""

import pytest

from src.dry_torch.trackers.csv import CSVDumper


class TestCsvDumper:
    """Tests for the HydraLink tracker with actual Hydra integration."""

    @pytest.fixture(autouse=True)
    def setup(self, start_experiment_event) -> None:
        """Setup test environment with actual hydra configuration."""
        self.tracker = CSVDumper()
        self.tracker.notify(start_experiment_event)

    def test_notify_with_epoch_metrics_new_file(self,
                                                epoch_metrics_event,
                                                sample_metrics):
        # Run the method
        self.tracker.notify(epoch_metrics_event)

        # Check the file was created with correct content
        csv_path = self.tracker._csv_path(str(epoch_metrics_event.model_name),
                                          str(epoch_metrics_event.source))
        assert csv_path.exists()

        epochs, metric_dict = self.tracker.read_csv(
            epoch_metrics_event.model_name,
            epoch_metrics_event.source,
        )

        assert epochs[0] == epoch_metrics_event.epoch
        for metric, value in metric_dict.items():
            assert sample_metrics[metric] == value[0]

        epoch_metrics_event.epoch += 1
        self.tracker.notify(epoch_metrics_event)

        # Check the file was created with correct content
        csv_path = self.tracker._csv_path(str(epoch_metrics_event.model_name),
                                          str(epoch_metrics_event.source))
        assert csv_path.exists()

        epochs, metric_dict = self.tracker.read_csv(
            epoch_metrics_event.model_name,
            epoch_metrics_event.source,
        )

        assert epochs[0] == epoch_metrics_event.epoch - 1  # previous event
        for metric, value in metric_dict.items():
            assert sample_metrics[metric] == value[0]

        assert epochs[1] == epoch_metrics_event.epoch
        for metric, value in metric_dict.items():
            assert sample_metrics[metric] == value[1]

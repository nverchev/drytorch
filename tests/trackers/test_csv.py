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
        csv_path = self.tracker.csv_path(str(epoch_metrics_event.model_name),
                                         str(epoch_metrics_event.source))
        assert csv_path.exists()

        columns, rows = self.tracker.read_csv(epoch_metrics_event.model_name,
                                              epoch_metrics_event.source)

        assert columns == ['Epoch', *sample_metrics.keys()]
        assert rows[0] == [epoch_metrics_event.epoch,
                           *sample_metrics.values()]

        self.tracker.notify(epoch_metrics_event)

        # Check the file was created with correct content
        csv_path = self.tracker.csv_path(str(epoch_metrics_event.model_name),
                                         str(epoch_metrics_event.source))
        assert csv_path.exists()

        columns, rows = self.tracker.read_csv(epoch_metrics_event.model_name,
                                              epoch_metrics_event.source)

        assert columns == ['Epoch', *sample_metrics.keys()]
        assert rows[0] == [epoch_metrics_event.epoch, *sample_metrics.values()]
        assert rows[1] == [epoch_metrics_event.epoch, *sample_metrics.values()]

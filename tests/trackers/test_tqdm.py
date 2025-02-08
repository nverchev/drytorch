"""Test suite for dry_torch progress bar functionality."""
import pytest
from io import StringIO

from dry_torch import log_events
from dry_torch.trackers.tqdm_logger import EpochBar, TrainingBar, TqdmLogger


class TestEpochBar:
    """Test suite for the EpochBar class."""

    @pytest.fixture(autouse=True)
    def setup(self, string_stream: StringIO) -> None:
        """
        Setup for each test.

        Args:
            string_stream: StringIO stream for capturing output
        """
        self.output = string_stream
        self.total_batches = 10
        self.dataset_size = 100
        self.bar = EpochBar(
            total=self.total_batches,
            num_samples=self.dataset_size,
            leave=False,
            out=self.output,
            desc="Training"
        )

    def test_initialization(self) -> None:
        """Test proper initialization of EpochBar."""
        assert self.bar.batch_size == self.dataset_size // self.total_batches
        assert self.bar.num_samples == self.dataset_size
        assert not self.bar.last_epoch
        assert self.bar.epoch_seen == 0

    def test_single_update(self, sample_metrics: dict[str, float]) -> None:
        """Test single update of the progress bar."""
        self.bar.update(sample_metrics)
        self.bar.pbar.refresh()
        output = self.output.getvalue()

        # Check if metrics are displayed
        for metric_name, value in sample_metrics.items():
            assert metric_name in output
            assert f"{value:.3e}" in output

        # Check if samples seen is updated
        assert self.bar.epoch_seen == self.bar.batch_size
        assert "Samples" in output

    def test_complete_epoch(self, sample_metrics: dict[str, float]) -> None:
        """Test progress bar behavior when epoch completes."""
        # Update until completion
        for _ in range(self.total_batches):
            self.bar.update(sample_metrics)

        assert self.bar.last_epoch
        assert self.bar.epoch_seen == self.dataset_size

        # Check if bar is closed
        assert self.bar.pbar.disable


class TestTrainingBar:
    """Test suite for the TrainingBar class."""

    @pytest.fixture(autouse=True)
    def setup(self, string_stream: StringIO) -> None:
        """
        Setup for each test.

        Args:
            string_stream: StringIO stream for capturing output
        """
        self.output = string_stream
        self.start_epoch = 0
        self.end_epoch = 10
        self.bar = TrainingBar(
            start_epoch=self.start_epoch,
            end_epoch=self.end_epoch,
            out=self.output,
            disable=False
        )

    def test_initialization(self) -> None:
        """Test proper initialization of TrainingBar."""
        assert self.bar.start_epoch == self.start_epoch
        assert self.bar.end_epoch == self.end_epoch

    def test_update(self) -> None:
        """Test updating the training progress bar."""
        current_epoch = 5
        self.bar.update(current_epoch)
        output = self.output.getvalue()

        assert f"Epoch: {current_epoch} / {self.end_epoch}" in output

    def test_close(self) -> None:
        """Test closing the training bar."""
        self.bar.close()
        assert self.bar.pbar.disable


class TestTqdmLogger:
    """Test suite for the TqdmLogger class."""

    @pytest.fixture(autouse=True)
    def setup(self, string_stream: StringIO) -> None:
        """
        Setup for each test.

        Args:
            string_stream: StringIO stream for capturing output
        """
        self.output = string_stream
        self.logger = TqdmLogger(
            leave=False,
            enable_training_bar=True,
            out=self.output
        )

    def test_iterate_batch_event(
            self,
            iterate_batch_event: log_events.IterateBatch
    ) -> None:
        """Test handling of IterateBatch event."""
        self.logger.notify(iterate_batch_event)

        # Check if update callback was added
        assert len(iterate_batch_event.push_updates) == 1

        # Try the update callback
        iterate_batch_event.push_updates[0]({"loss": 0.5})
        output = self.output.getvalue()
        assert iterate_batch_event.source in output

    def test_start_training_event(
            self,
            start_training_event: log_events.StartTraining,
    ) -> None:
        """Test handling of StartTraining event."""
        self.logger.notify(start_training_event)
        training_bar = self.logger.training_bar

        assert training_bar is not None
        assert training_bar.start_epoch == start_training_event.start_epoch
        assert training_bar.end_epoch == start_training_event.end_epoch

    def test_start_epoch_event(
            self,
            start_training_event: log_events.StartTraining,
            start_epoch_event: log_events.StartEpoch,
    ) -> None:
        """Test handling of StartEpoch event with active training bar."""
        # First create training bar
        self.logger.notify(start_training_event)

        # Then notify epoch start
        self.logger.notify(start_epoch_event)

        output = self.output.getvalue()
        assert f"Epoch: {start_epoch_event.epoch}" in output

    def test_disabled_training_bar(
            self,
            start_training_event: log_events.StartTraining,
    ) -> None:
        """Test TqdmLogger with disabled training bar."""
        logger = TqdmLogger(enable_training_bar=False)
        logger.notify(start_training_event)

        assert logger.training_bar is not None
        assert logger.training_bar.pbar.disable


@pytest.mark.parametrize("total,dataset_size,expected_batch_size", [
    (10, 100, 10),
    (4, 128, 32),
    (5, 1000, 200),
])
def test_epoch_bar_batch_size_calculation(
        string_stream: StringIO,
        total: int,
        dataset_size: int,
        expected_batch_size: int
) -> None:
    """
    Test batch size calculation for different configurations.

    Args:
        string_stream: StringIO stream for capturing output
        total: Total number of batches
        dataset_size: Total dataset size
        expected_batch_size: Expected calculated batch size
    """
    bar = EpochBar(
        total=total,
        num_samples=dataset_size,
        leave=False,
        out=string_stream,
        desc="Test"
    )
    assert bar.batch_size == expected_batch_size

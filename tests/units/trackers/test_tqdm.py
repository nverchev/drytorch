"""Tests for the "tqdm" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('tqdm'):
    pytest.skip('tqdm not available', allow_module_level=True)

from drytorch.trackers.tqdm import EpochBar, TqdmLogger, TrainingBar


class TestEpochBar:
    """Tests for the EpochBar class."""

    @pytest.fixture(autouse=True)
    def setup(self, string_stream) -> None:
        """Set up the tests."""
        self.stream = string_stream
        return

    @pytest.fixture
    def bar(self) -> EpochBar:
        """Set up the instance."""
        bar = EpochBar(
            num_iter=10,
            batch_size=32,
            num_samples=312,
            leave=False,
            file=self.stream,
            desc='Training',
        )
        return bar

    def test_initialization(self, bar) -> None:
        """Test instance attributes."""
        assert bar.pbar.desc

    def test_single_update(self, bar, example_named_metrics) -> None:
        """Test a single update of the progress bar."""
        bar.update(example_named_metrics)
        bar.pbar.refresh()
        output = self.stream.getvalue()
        for metric_name, value in example_named_metrics.items():
            assert metric_name in output
            assert f'{value:.3e}' in output

        assert bar._epoch_seen == bar._batch_size

    def test_complete_epoch(self, bar, example_named_metrics) -> None:
        """Test progress bar behavior when the epoch completes."""
        for _ in range(bar._num_iter):
            bar.update(example_named_metrics)

        assert bar._epoch_seen == bar._num_samples
        assert bar.pbar.disable


class TestTrainingBar:
    """Tests for the EpochBar class."""

    @pytest.fixture(autouse=True)
    def setup(self, string_stream) -> None:
        """Set up the tests."""
        self.stream = string_stream
        return

    @pytest.fixture
    def bar(self) -> TrainingBar:
        """Set up the instance."""
        bar = TrainingBar(
            start_epoch=0,
            end_epoch=12,
            file=self.stream,
            leave=False,
        )
        return bar

    def test_initialization(self, bar) -> None:
        """Test instance attributes."""
        assert bar.pbar.desc

    def test_update(self, bar) -> None:
        """Test updating the training progress bar."""
        current_epoch = 5
        bar.update(current_epoch)
        output = self.stream.getvalue()
        assert f'Epoch: {current_epoch} / {bar._end_epoch}' in output


class TestTqdmLogger:
    """Tests for the TqdmLogger class."""

    @pytest.fixture(autouse=True)
    def setup(self, string_stream) -> None:
        """Set up the tests."""
        self.stream = string_stream
        return

    @pytest.fixture
    def tracker(self) -> TqdmLogger:
        """Set up the instance."""
        return TqdmLogger(file=self.stream)

    @pytest.fixture
    def tracker_with_double_bar(self) -> TqdmLogger:
        """Set up the instance."""
        return TqdmLogger(enable_training_bar=True, file=self.stream)

    def test_cleanup(self, tracker):
        """Test correct clean up."""
        tracker.clean_up()
        assert tracker._training_bar is None
        assert tracker._epoch_bar is None

    def test_iterate_batch_event(
        self,
        tracker,
        iterate_batch_mock_event,
    ) -> None:
        """Test handling of the IterateBatch event."""
        tracker.notify(iterate_batch_mock_event)
        assert len(iterate_batch_mock_event.push_updates) == 1
        iterate_batch_mock_event.push_updates[0]({'loss': 0.5})
        output = self.stream.getvalue()
        assert iterate_batch_mock_event.source_name in output

    def test_start_training_event(
        self,
        tracker_with_double_bar,
        start_training_mock_event,
    ) -> None:
        """Test handling of the StartTraining event."""
        tracker_with_double_bar.notify(start_training_mock_event)
        assert tracker_with_double_bar._training_bar is not None

    def test_start_epoch_event(
        self,
        tracker_with_double_bar,
        start_training_mock_event,
        start_epoch_mock_event,
    ) -> None:
        """Test handling of StartEpoch event with active training bar."""
        tracker_with_double_bar.notify(start_training_mock_event)
        tracker_with_double_bar.notify(start_epoch_mock_event)
        output = self.stream.getvalue()
        assert f'Epoch: {start_epoch_mock_event.epoch}' in output

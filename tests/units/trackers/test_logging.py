"""Tests for the tracker.logging module."""

import pytest

import io
import logging
from typing import Generator

from dry_torch import log_events
from dry_torch.trackers.logging import INFO_LEVELS
from dry_torch.trackers.logging import BuiltinLogger
from dry_torch.trackers.logging import DryTorchFilter
from dry_torch.trackers.logging import DryTorchFormatter
from dry_torch.trackers.logging import ProgressFormatter
from dry_torch.trackers.logging import enable_default_handler
from dry_torch.trackers.logging import enable_propagation
from dry_torch.trackers.logging import disable_default_handler
from dry_torch.trackers.logging import disable_propagation
from dry_torch.trackers.logging import get_verbosity
from dry_torch.trackers.logging import set_formatter


class SubStream(io.StringIO):
    """Mock class that has the name of stdout"""
    name = '<stdout>'


@pytest.fixture()
def logger() -> logging.Logger:
    """Fixture for the library logger."""
    return logging.getLogger('dry_torch')


@pytest.fixture()
def mock_stdout() -> SubStream:
    """Fixture for the library logger."""
    return SubStream()


@pytest.fixture
def stream_handler(string_stream) -> logging.StreamHandler:
    """StreamHandler with library formatter."""
    return logging.StreamHandler(string_stream)


@pytest.fixture
def stdout_mock_handler(mock_stdout,
                        string_stream) -> logging.StreamHandler:
    """Mocks a handler to stdout because pytest redirects the handler."""
    stdout_handler = logging.StreamHandler(mock_stdout)
    return stdout_handler


class TestBuiltinLogger:
    """ Test suite for the BuiltinLogger class."""

    @pytest.fixture(autouse=True)
    def setup(
            self,
            logger,
            string_stream,
            stream_handler,
    ) -> Generator[None, None, None]:
        """Sets up a logger with temporary configuration."""
        self.stream = string_stream
        original_handlers = logger.handlers.copy()
        original_level = logger.level
        logger.handlers.clear()
        logger.addHandler(stream_handler)
        logger.setLevel(INFO_LEVELS.internal)
        yield

        logger.handlers.clear()
        logger.handlers.extend(original_handlers)
        logger.setLevel(original_level)
        return

    @pytest.fixture
    def tracker(self) -> BuiltinLogger:
        """Sets up the logger with temporary configuration."""
        return BuiltinLogger()

    def test_start_training_event(
            self,
            tracker,
            start_training_mock_event: log_events.StartTraining,
    ) -> None:
        """Tests handling of StartTraining event."""
        start_training_mock_event.model_name = 'my_model'
        tracker.notify(start_training_mock_event)
        expected = f'Training my_model started.'
        assert expected in self.stream.getvalue()

    def test_end_training_event(
            self,
            tracker,
            end_training_mock_event,
    ) -> None:
        """Tests handling of EndTraining event."""
        tracker.notify(end_training_mock_event)
        assert 'Training ended.' in self.stream.getvalue()

    def test_start_epoch_event_with_final_epoch(
            self,
            tracker,
            start_epoch_mock_event,
    ) -> None:
        """Tests handling of StartEpoch event with final epoch specified."""
        start_epoch_mock_event.epoch = 4
        start_epoch_mock_event.end_epoch = 10
        tracker.notify(start_epoch_mock_event)
        expected = f'====> Epoch  4/10:'
        assert expected in self.stream.getvalue()

    def test_start_epoch_without_final_epoch(self,
                                             tracker,
                                             start_epoch_mock_event) -> None:
        """Tests handling of StartEpoch event without final epoch specified."""
        start_epoch_mock_event.epoch = 12
        start_epoch_mock_event.end_epoch = None
        tracker.notify(start_epoch_mock_event)
        assert f'====> Epoch 12:' in self.stream.getvalue()

    def test_save_model_event(
            self,
            tracker,
            save_model_mock_event,
    ) -> None:
        """Tests handling of SaveModel event."""
        save_model_mock_event.model_name = 'my_model'
        save_model_mock_event.definition = 'weights'
        save_model_mock_event.location = 'folder'
        tracker.notify(save_model_mock_event)
        expected = f'Saving my_model weights in: folder.'
        assert expected in self.stream.getvalue()

    def test_load_model_event(
            self,
            tracker,
            load_model_mock_event,
    ) -> None:
        """Tests handling of LoadModel event."""
        load_model_mock_event.model_name = 'my_model'
        load_model_mock_event.definition = 'weights'
        load_model_mock_event.location = 'folder'
        load_model_mock_event.epoch = 3
        tracker.notify(load_model_mock_event)
        expected = f'Loading my_model weights at epoch 3.'
        assert expected in self.stream.getvalue()

    def test_test_event(self,
                        tracker,
                        start_test_mock_event) -> None:
        """Tests handling of Test event."""
        start_test_mock_event.model_name = 'my_model'
        tracker.notify(start_test_mock_event)
        assert f'Testing my_model started.' in self.stream.getvalue()

    def test_final_metrics_event(
            self,
            tracker,
            epoch_metrics_mock_event,
    ) -> None:
        """Tests handling of FinalMetrics event."""
        tracker.notify(epoch_metrics_mock_event)
        output = self.stream.getvalue()
        assert epoch_metrics_mock_event.source_name in output
        for metric_name in epoch_metrics_mock_event.metrics:
            assert metric_name in output

    def test_terminated_training_event(
            self,
            tracker,
            terminated_training_mock_event,
    ) -> None:
        """Tests handling of TerminatedTraining event."""
        terminated_training_mock_event.source_name = 'my_source'
        terminated_training_mock_event.model_name = 'my_model'
        terminated_training_mock_event.epoch = 10
        terminated_training_mock_event.reason = 'Test terminate'
        tracker.notify(terminated_training_mock_event)
        expected = 'my_source: Training my_model terminated at epoch 10. '
        expected += 'Reason: Test terminate'
        output = self.stream.getvalue()
        assert expected in output

    def test_update_learning_rate_event(
            self,
            tracker,
            update_learning_rate_mock_event,
    ) -> None:
        """Tests handling of UpdateLearningRate event."""
        update_learning_rate_mock_event.source_name = 'my_source'
        update_learning_rate_mock_event.model_name = 'my_model'
        update_learning_rate_mock_event.epoch = 10
        update_learning_rate_mock_event.scheduler_name = None
        update_learning_rate_mock_event.base_lr = None
        tracker.notify(update_learning_rate_mock_event)
        output = self.stream.getvalue()
        expected = 'Updated my_model optimizer at epoch 10.'
        assert expected in output
        update_learning_rate_mock_event.base_lr = 0.001
        tracker.notify(update_learning_rate_mock_event)
        output = self.stream.getvalue()
        expected += '\nNew learning rate: 0.001.'
        assert expected in output
        update_learning_rate_mock_event.scheduler_name = 'my_scheduler'
        tracker.notify(update_learning_rate_mock_event)
        output = self.stream.getvalue()
        expected += '\nNew scheduler: my_scheduler.'
        assert expected in output


@pytest.fixture()
def example_record() -> logging.LogRecord:
    """Set up the instance."""
    record = logging.LogRecord(
        name='testing',
        level=0,
        pathname='test.py',
        lineno=1,
        msg='Test message',
        args=(),
        exc_info=None
    )
    return record


class TestDryTorchFilter:
    """Test DryTorchFilter."""

    @pytest.fixture()
    def dry_filter(self) -> DryTorchFilter:
        """Set up the instance."""
        return DryTorchFilter()

    def test_filter(self, dry_filter, example_record) -> None:
        """Set up the instance."""
        assert dry_filter.filter(example_record)
        example_record.name = 'testing_dry_torch'
        assert not dry_filter.filter(example_record)


class TestDryTorchFormatter:
    """Test DryTorchFormatter."""

    @pytest.fixture()
    def formatter(self) -> DryTorchFormatter:
        """Set up the instance."""
        return DryTorchFormatter()

    def test_format_experiment_level(self, formatter, example_record) -> None:
        """Test formatting at experiment level."""
        example_record.levelno = INFO_LEVELS.experiment
        formatted = formatter.format(example_record)
        assert formatted.endswith('Test message\n')
        assert formatted.startswith('[')  # Check for timestamp

    def test_format_other_level(self, formatter, example_record) -> None:
        """Test formatting at epoch level."""
        formatted = formatter.format(example_record)
        assert formatted == 'Test message\n'


class TestProgressFormatter:
    """Tests ProgressFormatter."""

    @pytest.fixture()
    def formatter(self) -> ProgressFormatter:
        """Set up the instance."""
        return ProgressFormatter()

    def test_format_metric_level(self, formatter, example_record) -> None:
        """Test formatting at metric level."""
        example_record.levelno = INFO_LEVELS.metrics
        formatted = formatter.format(example_record)
        assert formatted.endswith('Test message\r')
        assert formatted.startswith('[')  # Check for timestamp

    def test_format_epoch_level(self, formatter, example_record) -> None:
        """Test formatting at epoch level."""
        example_record.levelno = INFO_LEVELS.epoch
        formatted = formatter.format(example_record)
        assert formatted.endswith('Test message')
        assert formatted.startswith('[')  # Check for timestamp


def test_set_formatter_style(stdout_mock_handler, logger) -> None:
    """Tests set formatter style."""
    logger.addHandler(stdout_mock_handler)
    set_formatter(style='dry_torch')
    assert isinstance(stdout_mock_handler.formatter, DryTorchFormatter)
    set_formatter(style='progress')
    assert isinstance(stdout_mock_handler.formatter, ProgressFormatter)


def test_enable_propagation(logger,
                            stdout_mock_handler,
                            mock_stdout,
                            stream_handler,
                            string_stream) -> None:
    """Tests enabling and disabling of log propagation."""
    logger.handlers.clear()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(stdout_mock_handler)
    enable_propagation()
    logger.error('test error 1')
    assert string_stream.getvalue() == 'test error 1\n'
    assert mock_stdout.getvalue() == ''
    disable_default_handler()
    disable_propagation()
    logger.error('test error 2')
    assert 'test error 2' not in string_stream.getvalue()


def test_enable_disable_default_handler(logger) -> None:
    """Tests enabling and disabling of default handler."""
    disable_default_handler()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)

    enable_default_handler()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.level == get_verbosity()
    assert logger.propagate is False

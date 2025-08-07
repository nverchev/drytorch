"""Tests for the "logging" module."""

import logging

from collections.abc import Generator

import pytest

from drytorch.core import log_events
from drytorch.trackers.logging import (
    INFO_LEVELS,
    BuiltinLogger,
    DryTorchFilter,
    DryTorchFormatter,
    ProgressFormatter,
    disable_default_handler,
    disable_propagation,
    enable_default_handler,
    enable_propagation,
    get_verbosity,
    set_formatter,
    set_verbosity,
)


@pytest.fixture
def stream_handler(string_stream) -> logging.StreamHandler:
    """StreamHandler with library formatter."""
    return logging.StreamHandler(string_stream)


@pytest.fixture()
def logger(stream_handler) -> logging.Logger:
    """Fixture for the library logger."""
    logger = logging.getLogger('drytorch')
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    return logger


@pytest.fixture()
def root_logger(string_stream) -> logging.Logger:
    """Fixture for the library logger."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(logging.StreamHandler(string_stream))
    return root_logger


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
        exc_info=None,
    )
    return record


class TestBuiltinLogger:
    """Test suite for the BuiltinLogger class."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        logger,
        string_stream,
        stream_handler,
    ) -> Generator[None, None, None]:
        """Set up a logger with temporary configuration."""
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
        """Set up the instance."""
        return BuiltinLogger()

    def test_start_training_event(
        self,
        tracker,
        start_training_mock_event: log_events.StartTrainingEvent,
    ) -> None:
        """Tests handling of StartTraining event."""
        start_training_mock_event.model_name = 'my_model'
        tracker.notify(start_training_mock_event)
        expected = 'Training my_model started.'
        assert expected in self.stream.getvalue()

    def test_end_training_event(
        self,
        tracker,
        end_training_mock_event,
    ) -> None:
        """Test handling of EndTraining event."""
        tracker.notify(end_training_mock_event)
        assert 'Training ended.' in self.stream.getvalue()

    def test_start_epoch_event_with_final_epoch(
        self,
        tracker,
        start_epoch_mock_event,
    ) -> None:
        """Test handling of StartEpoch event with final epoch specified."""
        start_epoch_mock_event.epoch = 4
        start_epoch_mock_event.end_epoch = 10
        tracker.notify(start_epoch_mock_event)
        expected = '====> Epoch  4/10:'
        assert expected in self.stream.getvalue()

    def test_start_epoch_without_final_epoch(
        self, tracker, start_epoch_mock_event
    ) -> None:
        """Test handling of StartEpoch event without final epoch specified."""
        start_epoch_mock_event.epoch = 12
        start_epoch_mock_event.end_epoch = None
        tracker.notify(start_epoch_mock_event)
        assert '====> Epoch 12:' in self.stream.getvalue()

    def test_save_model_event(
        self,
        tracker,
        save_model_mock_event,
    ) -> None:
        """Test handling of SaveModel event."""
        save_model_mock_event.model_name = 'my_model'
        save_model_mock_event.definition = 'weights'
        save_model_mock_event.location = 'folder'
        tracker.notify(save_model_mock_event)
        expected = 'Saving my_model weights in: folder.'
        assert expected in self.stream.getvalue()

    def test_load_model_event(
        self,
        tracker,
        load_model_mock_event,
    ) -> None:
        """Test handling of the LoadModel event."""
        load_model_mock_event.model_name = 'my_model'
        load_model_mock_event.definition = 'weights'
        load_model_mock_event.location = 'folder'
        load_model_mock_event.epoch = 3
        tracker.notify(load_model_mock_event)
        expected = 'Loading my_model weights at epoch 3.'
        assert expected in self.stream.getvalue()

    def test_test_event(self, tracker, start_test_mock_event) -> None:
        """Test handling of Test event."""
        start_test_mock_event.model_name = 'my_model'
        tracker.notify(start_test_mock_event)
        assert 'Testing my_model started.' in self.stream.getvalue()

    def test_final_metrics_event(
        self,
        tracker,
        epoch_metrics_mock_event,
    ) -> None:
        """Test handling of the FinalMetrics event."""
        epoch_metrics_mock_event.source_name = 'test_source'
        epoch_metrics_mock_event.metrics = {'loss': 1.2, 'accuracy': 0.81}
        tracker.notify(epoch_metrics_mock_event)
        expected = 'test_source:    loss=1.200000e+00   accuracy=8.100000e-01\n'
        assert self.stream.getvalue().expandtabs(4).endswith(expected)

    def test_terminated_training_event(
        self,
        tracker,
        terminated_training_mock_event,
    ) -> None:
        """Test handling of TerminatedTraining event."""
        terminated_training_mock_event.source_name = 'my_source'
        terminated_training_mock_event.model_name = 'my_model'
        terminated_training_mock_event.epoch = 10
        terminated_training_mock_event.reason = 'Test terminate'
        tracker.notify(terminated_training_mock_event)
        expected = 'Training my_model terminated at epoch 10. '
        expected += 'Reason: Test terminate'
        output = self.stream.getvalue()
        assert expected in output

    def test_update_learning_rate_event(
        self,
        tracker,
        update_learning_rate_mock_event,
    ) -> None:
        """Test handling of the UpdateLearningRate event."""
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
        expected += ' New learning rate: 0.001.'
        assert expected in output
        update_learning_rate_mock_event.scheduler_name = 'my_scheduler'
        tracker.notify(update_learning_rate_mock_event)
        output = self.stream.getvalue()
        expected += ' New scheduler: my_scheduler.'
        assert expected in output


class TestDryTorchFilter:
    """Test DryTorchFilter."""

    @pytest.fixture()
    def dry_filter(self) -> DryTorchFilter:
        """Set up the instance."""
        return DryTorchFilter()

    def test_filter(self, dry_filter, example_record) -> None:
        """Set up the instance."""
        assert dry_filter.filter(example_record)
        example_record.name = 'testing_drytorch'
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
        assert formatted.startswith('\r[')  # Check for timestamp

    def test_format_other_level(self, formatter, example_record) -> None:
        """Test formatting at epoch level."""
        formatted = formatter.format(example_record)
        assert formatted == '\rTest message\n'


class TestProgressFormatter:
    """Tests ProgressFormatter."""

    @pytest.fixture()
    def formatter(self) -> ProgressFormatter:
        """Set up the instance."""
        return ProgressFormatter()

    def test_format_metric_level(self, formatter, example_record) -> None:
        """Test formatting at metric level."""
        example_record.levelno = INFO_LEVELS.epoch
        formatted = formatter.format(example_record)
        assert formatted.endswith('...\r')

    def test_format_epoch_level(self, formatter, example_record) -> None:
        """Test formatting at epoch level."""
        example_record.levelno = INFO_LEVELS.model_state
        formatted = formatter.format(example_record)
        assert formatted.endswith('\r')


def test_disable_default_handler(logger) -> None:
    """Test disabling  default handler."""
    disable_default_handler()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_disable_propagation(logger, root_logger, string_stream) -> None:
    """Test disabling log propagation."""
    disable_default_handler()
    disable_propagation()
    logger.error('test error 3')
    assert not string_stream.getvalue()


def test_enable_default_handler(logger) -> None:
    """Test enabling default handler."""
    enable_default_handler()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_enable_propagation(logger, root_logger, string_stream) -> None:
    """Test enabling log propagation."""
    enable_propagation(False)
    logger.error('test error 1')
    assert string_stream.getvalue() == 'test error 1\ntest error 1\n'


def test_enable_propagation_with_deduplication(
    logger, root_logger, string_stream
) -> None:
    """Test enabling log propagation while deduplicating output."""
    enable_propagation()
    logger.error('test error 1')
    assert string_stream.getvalue() == 'test error 1\n'


def test_set_verbosity(logger) -> None:
    """Test setting verbosity level."""
    set_verbosity(INFO_LEVELS.test)
    assert INFO_LEVELS.test == get_verbosity()


def test_set_formatter_style(stream_handler, logger) -> None:
    """Test setting formatter style."""
    logger.addHandler(stream_handler)
    set_formatter(style='drytorch')
    assert isinstance(stream_handler.formatter, DryTorchFormatter)
    set_formatter(style='progress')
    assert isinstance(stream_handler.formatter, ProgressFormatter)

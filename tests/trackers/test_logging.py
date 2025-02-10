"""
Test suite for the dry_torch logging module.

This module contains pytest-based tests for verifying the functionality of:
- Custom formatters
- Logging level configurations
- Event handling
- Logger configuration management

The tests use pytest fixtures for setup and teardown, and include
comprehensive type hints and documentation.
"""

import pytest

import logging
import io
from typing import Generator

from dry_torch import log_events
from dry_torch.trackers.builtin_logger import BuiltinLogger, InfoFormatter
from dry_torch.trackers.builtin_logger import enable_default_handler
from dry_torch.trackers.builtin_logger import enable_propagation
from dry_torch.trackers.builtin_logger import disable_default_handler
from dry_torch.trackers.builtin_logger import disable_propagation
from dry_torch.trackers.builtin_logger import INFO_LEVELS

logger = logging.getLogger('dry_torch')


@pytest.fixture
def test_handler(string_stream: io.StringIO) -> logging.StreamHandler:
    """
    Creates a StreamHandler with InfoFormatter for testing.

    Args:
        string_stream: StringIO stream for capturing output

    Returns:
        logging.StreamHandler: Configured handler for testing
    """
    handler = logging.StreamHandler(string_stream)
    handler.setFormatter(InfoFormatter())
    return handler


@pytest.fixture
def configured_logger(
        test_handler: logging.StreamHandler,
) -> Generator[logging.Logger, None, None]:
    """
    Sets up the logger with temporary configuration .

    Args:
        test_handler: StreamHandler to use for testing

    Yields:
        logging.Logger: Configured logger instance
    """
    original_handlers = logger.handlers.copy()
    original_level = logger.level

    logger.handlers.clear()
    logger.addHandler(test_handler)
    logger.setLevel(INFO_LEVELS.metrics)

    yield logger

    logger.handlers.clear()
    logger.handlers.extend(original_handlers)
    logger.setLevel(original_level)


def test_info_formatter_training_level() -> None:
    """Tests the InfoFormatter's formatting for training level logs."""
    formatter = InfoFormatter()
    record = logging.LogRecord(
        name='test',
        level=INFO_LEVELS.training,
        pathname='test.py',
        lineno=1,
        msg='Test message',
        args=(),
        exc_info=None
    )
    record.levelno = INFO_LEVELS.training

    formatted = formatter.format(record)
    assert formatted.endswith("Test message\n")
    assert "[" in formatted  # Check for timestamp



class TestBuiltinLogger:
    """ Test suite for the BuiltinLogger class."""

    @pytest.fixture(autouse=True)
    def setup(self,
              configured_logger: logging.Logger,
              string_stream: io.StringIO) -> None:
        """
        Automatically setup test environment before each test.

        Args:
            configured_logger: Pre-configured logger instance
            string_stream: StringIO stream for capturing output
        """
        self.logger = BuiltinLogger()
        self.stream = string_stream

    def test_start_training_event(
            self,
            start_training_event: log_events.StartTraining,
    ) -> None:
        """Tests handling of StartTraining event."""
        self.logger.notify(start_training_event)
        expected = f'Training {start_training_event.model_name} started.'
        assert expected in self.stream.getvalue()

    def test_end_training_event(
            self,
            end_training_event: log_events.EndTraining,
    ) -> None:
        """Tests handling of EndTraining event."""
        self.logger.notify(end_training_event)
        assert 'Training ended.' in self.stream.getvalue()

    def test_start_epoch_event_with_final_epoch(
            self,
            start_epoch_event: log_events.StartEpoch,
    ) -> None:
        """Tests handling of StartEpoch event with final epoch specified."""
        self.logger.notify(start_epoch_event)
        start = start_epoch_event.epoch
        end = start_epoch_event.final_epoch
        expected = f'====> Epoch   {start}/{end}:'
        assert expected in self.stream.getvalue()

    def test_start_epoch_without_final_epoch(self) -> None:
        """Tests handling of StartEpoch event without final epoch specified."""
        event = log_events.StartEpoch(epoch=1, final_epoch=None)
        self.logger.notify(event)
        assert '====> Epoch 1:' in self.stream.getvalue()

    def test_save_model_event(
            self,
            save_model_event: log_events.SaveModel,
    ) -> None:
        """Tests handling of SaveModel event."""
        self.logger.notify(save_model_event)
        model_name = save_model_event.model_name
        definition = save_model_event.definition.capitalize()
        location = save_model_event.location
        expected = f'Saving {model_name} {definition} in: {location}.'
        assert expected in self.stream.getvalue()

    def test_load_model_event(
            self,
            load_model_event: log_events.LoadModel,
    ) -> None:
        """Tests handling of LoadModel event."""
        self.logger.notify(load_model_event)
        model_name = load_model_event.model_name
        definition = load_model_event.definition.capitalize()
        epoch = load_model_event.epoch
        expected = f'Loading {model_name} {definition} at epoch {epoch}.'
        assert expected in self.stream.getvalue()

    def test_test_event(self, test_event: log_events.Test) -> None:
        """Tests handling of Test event."""
        self.logger.notify(test_event)
        model_name = test_event.model_name
        assert f'Testing {model_name} started.' in self.stream.getvalue()

    def test_final_metrics_event(
            self,
            final_metrics_event: log_events.FinalMetrics,
    ) -> None:
        """Tests handling of FinalMetrics event."""
        self.logger.notify(final_metrics_event)
        output = self.stream.getvalue()
        assert final_metrics_event.source in output
        for metric_name in final_metrics_event.metrics:
            assert metric_name in output

    def test_terminated_training_event(
            self,
            terminated_training_event: log_events.TerminatedTraining,
    ) -> None:
        """Tests handling of TerminatedTraining event."""
        self.logger.notify(terminated_training_event)
        output = self.stream.getvalue()
        assert terminated_training_event.source in output
        assert terminated_training_event.model_name in output
        assert str(terminated_training_event.epoch) in output
        assert terminated_training_event.reason in output

    def test_update_learning_rate_event(
            self,
            update_learning_rate_event: log_events.UpdateLearningRate,
    ) -> None:
        """Tests handling of UpdateLearningRate event."""
        self.logger.notify(update_learning_rate_event)
        output = self.stream.getvalue()
        assert update_learning_rate_event.source in output
        assert update_learning_rate_event.model_name in output
        assert str(update_learning_rate_event.epoch) in output
        if update_learning_rate_event.scheduler_name:
            assert update_learning_rate_event.scheduler_name in output


def test_enable_disable_propagation(configured_logger: logging.Logger) -> None:
    """
    Tests enabling and disabling of log propagation.

    Args:
        configured_logger: Pre-configured logger instance
    """
    enable_propagation()
    assert logger.propagate is True

    disable_propagation()
    assert logger.propagate is False


def test_enable_disable_default_handler(
        configured_logger: logging.Logger,
) -> None:
    """
    Tests enabling and disabling of default handler.

    Args:
        configured_logger: Pre-configured logger instance
    """
    disable_default_handler()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)

    enable_default_handler()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.level == INFO_LEVELS.metrics
    assert logger.propagate is False

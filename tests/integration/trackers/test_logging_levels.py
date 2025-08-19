"""Tests the events' log filter for all the logging levels."""

import logging
import pathlib

from collections.abc import Generator

import pytest

from drytorch.core import log_events
from drytorch.trackers.logging import INFO_LEVELS, BuiltinLogger, set_verbosity


expected_path_folder = pathlib.Path(__file__).parent / 'expected_logs'
expected_path_dict = {
    INFO_LEVELS.internal: expected_path_folder / 'internal_logging.txt',
    INFO_LEVELS.metrics: expected_path_folder / 'metrics_logging.txt',
    INFO_LEVELS.epoch: expected_path_folder / 'epoch_logging.txt',
    INFO_LEVELS.model_state: expected_path_folder / 'model_state_logging.txt',
    INFO_LEVELS.experiment: expected_path_folder / 'experiment_logging.txt',
    INFO_LEVELS.training: expected_path_folder / 'training_logging.txt',
    INFO_LEVELS.test: expected_path_folder / 'test_logging.txt',
}


@pytest.fixture
def logger() -> logging.Logger:
    """Fixture for the library logger."""
    return logging.getLogger('drytorch')


@pytest.fixture
def stream_handler(string_stream) -> logging.StreamHandler:
    """StreamHandler with library formatter."""
    return logging.StreamHandler(string_stream)


@pytest.fixture
def setup(
    request,
    logger,
    string_stream,
    stream_handler,
) -> Generator[None, None, None]:
    """Set up a logger with temporary configuration."""
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    yield

    logger.handlers.clear()
    logger.handlers.extend(original_handlers)
    logger.setLevel(original_level)
    return


@pytest.fixture
def event_workflow(
    start_experiment_event,
    model_registration_event,
    load_model_event,
    start_training_event,
    actor_registration_event,
    start_epoch_event,
    iterate_batch_event,
    metrics_event,
    end_epoch_event,
    update_learning_rate_event,
    terminated_training_event,
    end_training_event,
    start_test_event,
    end_test_event,
    save_model_event,
    stop_experiment_event,
) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    event_tuple = (
        start_experiment_event,
        model_registration_event,
        load_model_event,
        start_training_event,
        actor_registration_event,
        start_epoch_event,
        iterate_batch_event,
        metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        end_test_event,
        save_model_event,
        stop_experiment_event,
    )
    return event_tuple


@pytest.mark.parametrize('info_level', list(INFO_LEVELS))
def test_all_events(setup, event_workflow, string_stream, info_level):
    """Test unformatted logs for every level."""
    set_verbosity(info_level)
    tracker = BuiltinLogger()
    for event in event_workflow:
        tracker.notify(event)
    with expected_path_dict[info_level].open() as file:
        expected = file.read().strip()
    actual = string_stream.getvalue().strip().expandtabs(4)
    assert actual == expected

"""Tests the logs of a typical session for all the logging levels."""

import pytest

import logging
import re
from typing import Generator

from dry_torch import log_events
from dry_torch.trackers.tqdm import TqdmLogger
from dry_torch.trackers.logging import BuiltinLogger
from dry_torch.trackers.logging import INFO_LEVELS
from dry_torch.trackers.logging import set_verbosity

expected_metrics_level = """
====> Epoch   5/100:
    test_source: 100%|##########| 5/5, 00:00<00:00, Samples=1600, loss=4.560e-01, accuracy=8.920e-01
     test_model: 	loss=4.560000e-01	accuracy=8.920000e-01
====> Epoch   5/100:
    test_source: 100%|##########| 5/5, 00:00<00:00, Samples=1600, loss=4.560e-01, accuracy=8.920e-01
     test_model: 	loss=4.560000e-01	accuracy=8.920000e-01
""".strip()

expected_epoch_level = """
====> Epoch   5/100:
    test_source: 100%|##########| 5/5, 00:00<00:00, Samples=1600, loss=4.560e-01, accuracy=8.920e-01
====> Epoch   5/100:
    test_source: 100%|##########| 5/5, 00:00<00:00, Samples=1600, loss=4.560e-01, accuracy=8.920e-01
""".strip()

expected_param_update_level = """ 
    test_source: 100%|##########| 5/5, 00:00<00:00, Samples=1600, loss=4.560e-01, accuracy=8.920e-01
    test_source: 100%|##########| 5/5, 00:00<00:00, Samples=1600, loss=4.560e-01, accuracy=8.920e-01
""".strip()

expected_dict = {
    INFO_LEVELS.metrics: expected_metrics_level,
    INFO_LEVELS.epoch: expected_epoch_level,
    INFO_LEVELS.param_update: expected_param_update_level,
}


@pytest.fixture
def logger() -> logging.Logger:
    """Fixture for the library logger."""
    return logging.getLogger('dry_torch')


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
        start_epoch_event,
        call_model_event,
        iterate_batch_event,
        epoch_metrics_event,
) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    event_tuple = (
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
    )
    return event_tuple


selected_info_levels = [
    INFO_LEVELS.metrics,
    INFO_LEVELS.epoch,
    INFO_LEVELS.param_update,
]


def _strip_color(text):
    """Removes color to test for equality."""
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def _remove_carriage_return(text):
    """Removes carriage returns to test for equality."""
    text = _strip_color(text)
    return '\n'.join((sentence.rsplit('\r', maxsplit=1)[-1]
                      for sentence in text.split('\n')))


@pytest.mark.parametrize('info_level', selected_info_levels)
def test_typical_workflow(setup,
                          example_named_metrics,
                          event_workflow,
                          string_stream,
                          info_level):
    set_verbosity(info_level)
    logging_tracker = BuiltinLogger()
    tqdm_tracker = TqdmLogger(leave=True, out=string_stream)
    for event in event_workflow:
        logging_tracker.notify(event)
        tqdm_tracker.notify(event)
        if isinstance(event, log_events.IterateBatch):
            for _ in range(event.num_iter):
                event.update(example_named_metrics)
    actual = _strip_color(string_stream.getvalue())
    actual = _remove_carriage_return(actual)
    actual = actual.strip()
    assert actual == expected_dict[info_level]

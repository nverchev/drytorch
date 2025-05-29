"""Tests that tqdm functioning when using training and epoch bars."""
import pytest

import datetime
import io
import logging
import pathlib
import re
from typing import Generator

from dry_torch import tracking
from dry_torch import log_events
from dry_torch.trackers.tqdm import TqdmLogger
from dry_torch.trackers.tqdm import EpochBar
from dry_torch.trackers.tqdm import TrainingBar
from dry_torch.trackers.logging import BuiltinLogger
from dry_torch.trackers.logging import INFO_LEVELS
from dry_torch.trackers.logging import set_verbosity
from dry_torch.trackers.logging import DryTorchFormatter
from dry_torch.trackers.logging import ProgressFormatter

expected_path_folder = pathlib.Path() / 'expected_logs'


class MockStdout(io.StringIO):
    """Class that overwrites StringIO name."""
    name = '<stdout>'


@pytest.fixture()
def logger() -> logging.Logger:
    """Fixture for the library logger."""
    return logging.getLogger('dry_torch')


@pytest.fixture
def mock_stdout() -> MockStdout:
    """String stream that has the name of the stdout stream."""
    return MockStdout()


@pytest.fixture
def mock_stdout_handler(mock_stdout) -> logging.StreamHandler:
    """Handler that uses the stdout mock as stream."""
    stdout_handler = logging.StreamHandler(mock_stdout)
    return stdout_handler


@pytest.fixture
def setup(
        monkeypatch,
        logger,
        mock_stdout_handler,
) -> Generator[None, None, None]:
    """Set up a logger with temporary configuration."""
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    logger.handlers.clear()
    mock_stdout_handler.setFormatter(DryTorchFormatter())
    mock_stdout_handler.terminator = ''
    logger.addHandler(mock_stdout_handler)
    # remove elapsed time prints for reproducibility
    epoch_bar_fmt = EpochBar.fmt
    EpochBar.fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}'
    training_bar_fmt = TrainingBar.fmt
    TrainingBar.fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}'
    # fix timestamp for reproducibility
    monkeypatch.setattr(logging.Formatter, 'formatTime', _mock_format_time)
    yield

    logger.handlers.clear()
    logger.handlers.extend(original_handlers)
    logger.setLevel(original_level)
    EpochBar.fmt = epoch_bar_fmt
    TrainingBar.fmt = training_bar_fmt
    return


@pytest.fixture
def event_workflow(
        start_training_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        save_model_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        end_test_event,
) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    event_tuple = (
        start_training_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        save_model_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        iterate_batch_event,
        epoch_metrics_event,
        end_test_event,
    )
    return event_tuple


def test_standard_trackers(setup,
                           example_named_metrics,
                           event_workflow,
                           mock_stdout):
    """Test stdout output on standard workflow."""
    set_verbosity(INFO_LEVELS.epoch)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    trackers.append(TqdmLogger(file=mock_stdout))
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    expected_path = expected_path_folder / 'standard_trackers.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(mock_stdout) == expected


def test_standard_trackers_no_tqdm(setup,
                                   example_named_metrics,
                                   event_workflow,
                                   mock_stdout):
    """Test stdout output on standard workflow when tqdm is not available."""
    set_verbosity(INFO_LEVELS.metrics)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    expected_path = expected_path_folder / 'standard_trackers_no_tqdm.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(mock_stdout) == expected


def test_tuning_trackers(setup,
                         example_named_metrics,
                         event_workflow,
                         mock_stdout):
    """Test stdout output on tuning workflow."""
    set_verbosity(INFO_LEVELS.training)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    trackers.append(TqdmLogger(enable_training_bar=True, file=mock_stdout))
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    expected_path = expected_path_folder / 'tuning_trackers.txt'
    # some output is overwritten
    assert '\r' in mock_stdout.getvalue()
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(mock_stdout) == expected


def test_tuning_trackers_no_training_bar(setup,
                                         example_named_metrics,
                                         event_workflow,
                                         mock_stdout):
    """Test stdout output on tuning workflow without the training bar."""
    set_verbosity(INFO_LEVELS.training)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    trackers.append(TqdmLogger(file=mock_stdout))
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    # some output is overwritten
    assert '\r' in mock_stdout.getvalue()
    expected_path = expected_path_folder / 'tuning_trackers_no_training_bar.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(mock_stdout) == expected


def test_tuning_trackers_no_tqdm(setup,
                                 mock_stdout_handler,
                                 example_named_metrics,
                                 event_workflow,
                                 mock_stdout):
    """Test stdout output on tuning workflow when tqdm is not available."""
    set_verbosity(INFO_LEVELS.metrics)
    mock_stdout_handler.setFormatter(ProgressFormatter())
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    # some output is overwritten
    assert '\r' in mock_stdout.getvalue()
    expected_path = expected_path_folder / 'tuning_trackers_no_tqdm.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(mock_stdout) == expected


def _mock_format_time(*_, **__):
    fixed_time = datetime.datetime(2024, 1, 1, 12)
    return fixed_time.strftime('%Y-%m-%d %H:%M:%S')


def _notify_workflow(event_workflow: tuple[log_events.Event, ...],
                     trackers: list[tracking.Tracker],
                     example_named_metrics: dict[str, float]) -> None:
    for event in event_workflow:
        for tracker in trackers:
            tracker.notify(event)
            if isinstance(event, log_events.IterateBatch):
                for _ in range(event.num_iter):
                    event.update(example_named_metrics)
                event.push_updates.clear()  # necessary to reinitialize

    return


def _get_cleaned_value(mock_stdout: io.StringIO) -> str:
    text = mock_stdout.getvalue()
    text = _remove_up(text)
    text = _remove_carriage_return(text)
    text = _strip_color(text)
    return text.strip().expandtabs(4)


def _remove_carriage_return(text: str) -> str:
    """Remove lines ending with carriage returns."""
    text = _strip_color(text)
    return '\n'.join((line.rsplit('\r', maxsplit=1)[-1]
                      for line in text.split('\n')))


def _strip_color(text: str) -> str:
    """Remove color to test for equality."""
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def _remove_up(text: str) -> str:
    text = text.replace('\x1b[A\n', '')  # removes up and new line
    text_split = text.split('\n')
    new_split = list[str]()
    for line, next_line in zip(text_split, text_split[1:]):
        if '\x1b[A\r' not in next_line:
            new_split.append(line)
    new_split.append(text_split[-1])
    return '\n'.join(new_split)

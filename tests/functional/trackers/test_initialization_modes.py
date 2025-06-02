"""Tests the initialization modes whether tqdm is present or not."""

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
from dry_torch.trackers.logging import enable_default_handler
from dry_torch.trackers.logging import INFO_LEVELS
from dry_torch.trackers.logging import set_formatter
from dry_torch.trackers.logging import set_verbosity

expected_path_folder = pathlib.Path() / 'expected_logs'


@pytest.fixture()
def logger() -> logging.Logger:
    """Fixture for the library logger."""
    return logging.getLogger('dry_torch')


@pytest.fixture(autouse=True, )
def setup(
        monkeypatch,
        logger,
        string_stream,
) -> Generator[None, None, None]:
    """Set up a logger with temporary configuration."""

    def _mock_format_time(*_, **__):
        fixed_time = datetime.datetime(2024, 1, 1, 12)
        return fixed_time.strftime('%Y-%m-%d %H:%M:%S')

    # fix timestamp for reproducibility
    monkeypatch.setattr(logging.Formatter, 'formatTime', _mock_format_time)
    # remove elapsed time prints for reproducibility
    epoch_bar_fmt = EpochBar.fmt
    EpochBar.fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}'
    training_bar_fmt = TrainingBar.fmt
    TrainingBar.fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt}'
    # FIXME: reroute stderr / stdout instead
    enable_default_handler(stream=string_stream)
    yield

    enable_default_handler()
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


def test_standard_mode(example_named_metrics,
                       event_workflow,
                       string_stream):
    """Test standard mode on typical workflow."""
    set_verbosity(INFO_LEVELS.epoch)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    trackers.append(TqdmLogger(file=string_stream))
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    expected_path = expected_path_folder / 'standard_trackers.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(string_stream) == expected


def test_standard_mode_no_tqdm(example_named_metrics,
                               event_workflow,
                               string_stream):
    """Test standard mode on typical workflow when tqdm is not available."""
    set_verbosity(INFO_LEVELS.metrics)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    expected_path = expected_path_folder / 'standard_trackers_no_tqdm.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(string_stream) == expected


def test_hydra_mode(example_named_metrics,
                    event_workflow,
                    string_stream):
    """Test hydra mode on typical workflow."""
    set_verbosity(INFO_LEVELS.metrics)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    trackers.append(TqdmLogger(file=string_stream, leave=False))
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    # some output is overwritten
    assert '\r' in string_stream.getvalue()
    expected_path = expected_path_folder / 'standard_trackers_no_tqdm.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(string_stream) == expected


def test_tuning_mode(example_named_metrics,
                     event_workflow,
                     string_stream):
    """Test tuning mode on typical workflow."""
    set_verbosity(INFO_LEVELS.training)
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    trackers.append(TqdmLogger(enable_training_bar=True, file=string_stream))
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    expected_path = expected_path_folder / 'tuning_trackers.txt'
    # some output is overwritten
    assert '\r' in string_stream.getvalue()
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(string_stream) == expected


def test_tuning_mode_no_tqdm(example_named_metrics,
                             event_workflow,
                             string_stream):
    """Test tuning mode on typical workflow when tqdm is not available."""
    set_verbosity(INFO_LEVELS.epoch)
    set_formatter('progress')
    trackers = list[tracking.Tracker]()
    trackers.append(BuiltinLogger())
    _notify_workflow(event_workflow, trackers, example_named_metrics)
    # some output is overwritten
    assert '\r' in string_stream.getvalue()
    expected_path = expected_path_folder / 'tuning_trackers_no_tqdm.txt'
    with expected_path.open() as file:
        expected = file.read().strip()
    assert _get_cleaned_value(string_stream) == expected


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

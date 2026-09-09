"""Configuration module defining example events."""

import dataclasses
import pathlib

from collections.abc import Callable

import pytest

from drytorch.core import log_events
from tests.functional.trackers.conftest import (
    actor_registration_event,
    allow_event_creation_outside_scope,
    continue_experiment_event,
    end_epoch_event,
    end_test_event,
    end_training_event,
    event_workflow,
    iterate_batch_event,
    load_model_event,
    metrics_event,
    model_registration_event,
    pause_experiment_event,
    save_model_event,
    start_epoch_event,
    start_experiment_event,
    start_test_event,
    start_training_event,
    stop_experiment_event,
    string_stream,
    terminated_training_event,
    update_learning_rate_event,
)


_fixtures = (
    start_experiment_event,
    stop_experiment_event,
    pause_experiment_event,
    continue_experiment_event,
    model_registration_event,
    actor_registration_event,
    save_model_event,
    load_model_event,
    start_training_event,
    start_epoch_event,
    end_epoch_event,
    iterate_batch_event,
    terminated_training_event,
    end_training_event,
    start_test_event,
    end_test_event,
    metrics_event,
    update_learning_rate_event,
    string_stream,
    event_workflow,
    allow_event_creation_outside_scope,
)


@pytest.fixture
def run_b_id() -> str:
    """Provides a second distinct run identifier."""
    return 'run_b'


@pytest.fixture
def start_experiment_event_b(
    start_experiment_event: log_events.StartExperimentEvent,
    run_b_id: str,
) -> log_events.StartExperimentEvent:
    """Provides a StartExperiment event for a second run."""
    return dataclasses.replace(start_experiment_event, run_id=run_b_id)


@pytest.fixture
def stop_experiment_event_b(
    stop_experiment_event: log_events.StopExperimentEvent,
    run_b_id: str,
) -> log_events.StopExperimentEvent:
    """Provides a StopExperiment event for a second run."""
    return dataclasses.replace(stop_experiment_event, run_id=run_b_id)


@pytest.fixture
def metrics_event_a1(
    metrics_event: log_events.MetricEvent,
) -> log_events.MetricEvent:
    """Provides initial metrics event for run A."""
    return dataclasses.replace(metrics_event, epoch=1, metrics={'loss': 0.1})


@pytest.fixture
def metrics_event_b(
    metrics_event: log_events.MetricEvent,
) -> log_events.MetricEvent:
    """Provides distinct metrics event for run B."""
    return dataclasses.replace(metrics_event, epoch=1, metrics={'loss': 0.9})


@pytest.fixture
def metrics_event_a2(
    metrics_event: log_events.MetricEvent,
) -> log_events.MetricEvent:
    """Provides continued metrics event for run A."""
    return dataclasses.replace(metrics_event, epoch=2, metrics={'loss': 0.05})


@pytest.fixture
def interleaved_workflow(
    start_experiment_event: log_events.StartExperimentEvent,
    metrics_event_a1: log_events.MetricEvent,
    pause_experiment_event: log_events.PauseExperimentEvent,
    start_experiment_event_b: log_events.StartExperimentEvent,
    metrics_event_b: log_events.MetricEvent,
    stop_experiment_event_b: log_events.StopExperimentEvent,
    continue_experiment_event: log_events.ContinueExperimentEvent,
    metrics_event_a2: log_events.MetricEvent,
    stop_experiment_event: log_events.StopExperimentEvent,
) -> tuple[log_events.Event, ...]:
    """Provides interleaved event sequence for two runs (A and B)."""
    return (
        start_experiment_event,
        metrics_event_a1,
        pause_experiment_event,
        start_experiment_event_b,
        metrics_event_b,
        stop_experiment_event_b,
        continue_experiment_event,
        metrics_event_a2,
        stop_experiment_event,
    )


def resolve_run_dir(
    base_dir: pathlib.Path, folder_name: str, exp_name: str, run_id: str
) -> pathlib.Path:
    """Resolve run directory independently of Dumper._get_run_dir."""
    exp_dir = base_dir / folder_name / exp_name
    if '@' in run_id:
        day, time = run_id.split('@', 1)
        return exp_dir / day / time

    return exp_dir / run_id


@pytest.fixture
def run_dir_resolver() -> Callable[[pathlib.Path, str, str, str], pathlib.Path]:
    """Provides a helper to compute tracker run directories."""
    return resolve_run_dir


@pytest.fixture
def run_a_dir(
    tmp_path: pathlib.Path, example_exp_name: str, example_run_id: str
) -> Callable[[str], pathlib.Path]:
    """Returns a function to get run A directory for a folder name."""

    def _dir(folder_name: str) -> pathlib.Path:
        return resolve_run_dir(
            tmp_path, folder_name, example_exp_name, example_run_id
        )

    return _dir


@pytest.fixture
def run_b_dir(
    tmp_path: pathlib.Path, example_exp_name: str, run_b_id: str
) -> Callable[[str], pathlib.Path]:
    """Returns a function to get run B directory for a folder name."""

    def _dir(folder_name: str) -> pathlib.Path:
        return resolve_run_dir(
            tmp_path, folder_name, example_exp_name, run_b_id
        )

    return _dir

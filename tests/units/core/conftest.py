"""Configuration module with mockups."""

import pathlib

import pytest

from drytorch import Experiment
from drytorch.core.experiment import Run


@pytest.fixture(scope='module')
def mock_run(session_mocker, tmpdir_factory, example_run_id) -> Run:
    """Fixture for a mock experiment."""
    mock_run = session_mocker.create_autospec(Run, instance=True)
    mock_run.metadata_manager = session_mocker.Mock()
    mock_run.metadata_manager.record_model_call = session_mocker.Mock()
    mock_run.metadata_manager.bind_model = session_mocker.Mock()
    mock_run.id = example_run_id
    return mock_run


@pytest.fixture(scope='module')
def mock_experiment(session_mocker, tmpdir_factory) -> Experiment:
    """Fixture for a mock experiment."""
    mock_experiment = session_mocker.create_autospec(Experiment, instance=True)
    mock_experiment.name = 'mock_experiment'
    mock_experiment.run_dir = pathlib.Path(tmpdir_factory.mktemp('experiments'))
    mock_experiment.previous_runs = []
    return mock_experiment

"""Configuration module with mockups."""

import pathlib

import pytest

from drytorch import Experiment


@pytest.fixture(scope='module')
def mock_experiment(session_mocker, tmpdir_factory) -> Experiment:
    """Fixture for a mock experiment."""
    mock_experiment = session_mocker.create_autospec(Experiment, instance=True)
    mock_experiment.name = 'mock_experiment'
    mock_experiment.par_dir = pathlib.Path(tmpdir_factory.mktemp('experiments'))
    mock_experiment._metadata_manager = session_mocker.Mock()
    mock_experiment._metadata_manager.record_model_call = session_mocker.Mock()
    mock_experiment._metadata_manager.register_model = session_mocker.Mock()
    return mock_experiment

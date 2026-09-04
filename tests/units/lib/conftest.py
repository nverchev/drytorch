"""Configuration module for tests in lib."""

import pathlib

import pytest

from drytorch.core.experimenting import Experiment


@pytest.fixture(autouse=True, scope='module')
def setup_module_experiment(module_mocker, tmpdir_factory) -> None:
    """Fixture for a mock experiment shared across units in lib."""
    mock_experiment = module_mocker.create_autospec(Experiment, instance=True)
    mock_experiment.name = 'mock_experiment'
    mock_experiment.par_dir = pathlib.Path(tmpdir_factory.mktemp('par_dir'))
    mock_experiment.run = module_mocker.Mock()
    mock_experiment.run.id = 'run_id'
    mock_experiment.run_dir = (
        mock_experiment.par_dir / 'checkpoints' / 'mock_experiment' / 'run_id'
    )
    module_mocker.patch(
        'drytorch.Experiment.get_current', return_value=mock_experiment
    )

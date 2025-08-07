"""Test for the "experiments" module."""
from typing import Generator

import pytest

from drytorch.core import exceptions, log_events

from drytorch.core.experiments import Experiment


class TestExperiment:
    """Test the Experiment class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        _ = mocker.patch.object(log_events, 'StartExperimentEvent')
        _ = mocker.patch.object(log_events, 'StopExperimentEvent')
        return

    @pytest.fixture()
    def config(self, tmp_path) -> object:
        """Set up a test config object."""
        return object()

    @pytest.fixture()
    def experiment(self, config, tmp_path) -> Experiment:
        """Set up an experiment."""
        experiment = Experiment(config, name='Experiment',  par_dir=tmp_path)
        return experiment

    def test_start_and_stop_experiment(
            self, config, experiment, tmp_path
    ) -> None:
        """Test starting and stopping an experiment."""
        with experiment:
            assert Experiment.current() is experiment
            assert Experiment.current().par_dir == tmp_path
            assert Experiment.get_config() is config

    def test_no_active_experiment_error(self, experiment) -> None:
        """Test that error is called when no experiment is active."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            # Experiment.current has been stored in experiment_current_original
            _ = experiment.current()

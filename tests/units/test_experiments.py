"""Test for the "experiments" module."""

import pytest

from drytorch import Experiment, exceptions, log_events
from tests.units.conftest import experiment_current_original


class TestExperiment:
    """Test the Experiment class."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path) -> None:
        """Set up an experiment."""
        self.par_dir = tmp_path
        self.experiment = Experiment[None]('Experiment', self.par_dir)
        self.experiment.current = experiment_current_original
        Experiment.current = experiment_current_original
        self.experiment.__class__.current = experiment_current_original
        return

    def test_start_and_stop_experiment(self, mocker):
        """Test starting and stopping an experiment."""
        mock_event_start = mocker.patch.object(log_events, 'StartExperiment')
        mock_event_stop = mocker.patch.object(log_events, 'StopExperiment')
        with self.experiment:
            path = self.par_dir / self.experiment.name
            mock_event_start.assert_called_once_with(self.experiment.name,
                                                     self.experiment.created_at,
                                                     path,
                                                     None)
        mock_event_stop.assert_called_once_with(self.experiment.name)

    def test_get_config_no_config_error(self):
        """Test NoConfigError is raised if config is None."""
        with self.experiment:
            with pytest.raises(exceptions.NoConfigurationError):
                Experiment.get_config()

    def test_no_active_experiment_error(self):
        """Test that error is called when no experiment is active."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            # Experiment.current has been stored in experiment_current_original
            _ = self.experiment.current()

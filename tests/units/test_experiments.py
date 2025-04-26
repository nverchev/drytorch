"""Test for the experiment module."""

import pytest

from dry_torch import Experiment, log_events, exceptions
from tests.units.conftest import experiment_current_original


class TestExperiment:
    """Test the Experiment class."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path) -> None:
        """Set up an experiment."""
        self.name = 'TestExperiment'
        self.par_dir = tmp_path
        self.experiment = Experiment[None](self.name, self.par_dir)
        setattr(self.experiment, 'current', experiment_current_original)
        setattr(Experiment,
                'current',
                experiment_current_original)
        setattr(self.experiment.__class__,
                'current',
                experiment_current_original)
        return

    def test_start_and_stop_experiment(self, mocker):
        """Test starting and stopping an experiment."""
        mock_event_start = mocker.patch.object(log_events, 'StartExperiment')
        mock_event_stop = mocker.patch.object(log_events, 'StopExperiment')
        with self.experiment:
            path = self.par_dir / format(self.name, 's')
            mock_event_start.assert_called_once_with(self.experiment.name,
                                                     self.experiment.created_at,
                                                     path,
                                                     None)
        mock_event_stop.assert_called_once_with(self.experiment.name)

    def test_get_config_no_config_error(self):
        """Test NoConfigError is raised if config is None."""
        with self.experiment:
            with pytest.raises(exceptions.NoConfigError):
                Experiment.get_config()

    def test_no_active_experiment_error(self):
        """Test that error is called when no experiment is active."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            # Experiment.current has been stored in experiment_current_original
            _ = self.experiment.current()

"""Test for the "experiments" module."""

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.experiments import Experiment, Run


class TestExperiment:
    """Test the Experiment class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        mocker.patch.object(log_events, 'StartExperimentEvent')
        mocker.patch.object(log_events, 'StopExperimentEvent')
        return

    @pytest.fixture()
    def config(self) -> object:
        """Set up a test config object."""
        return object()

    @pytest.fixture()
    def experiment(self, config, tmp_path) -> Experiment:
        """Set up an experiment."""
        return Experiment(config, name='Experiment', par_dir=tmp_path)

    def test_no_active_experiment_error(self, experiment) -> None:
        """Test that an error is raised when no experiment is active."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            Experiment.get_current()


class TestRun:
    """Test the Run class and its context management."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up mocks for event logging."""
        mocker.patch.object(log_events, 'StartExperimentEvent')
        mocker.patch.object(log_events, 'StopExperimentEvent')
        return

    @pytest.fixture()
    def config(self) -> object:
        """Set up a test config object."""
        return object()

    @pytest.fixture()
    def experiment(self, config, tmp_path) -> Experiment:
        """Set up an experiment."""
        return Experiment(config, name='Experiment', par_dir=tmp_path)

    @pytest.fixture()
    def run(self, experiment) -> Run:
        """Set up a run for an experiment."""
        return Run(experiment)

    def test_start_and_stop_run(
            self, run, experiment, config, tmp_path
        ) -> None:
        """Test starting and stopping a run using the context manager."""
        with run:
            assert Experiment.get_current() is experiment
            assert Experiment.get_current().par_dir == tmp_path
            assert Experiment.get_config() is config

        with pytest.raises(exceptions.NoActiveExperimentError):
            Experiment.get_current()

    def test_run_is_added_to_experiment_runs_list(self, experiment) -> None:
        """Test that a new run is added to the experiment's run list."""
        experiment.runs.clear()
        run1 = Run(experiment)
        run2 = Run(experiment)
        assert len(experiment.runs) == 2
        assert experiment.runs == [run1, run2]

    def test_nested_scope_error(self, run) -> None:
        """Test that an error is raised for nested runs."""
        with run:
            with pytest.raises(exceptions.NestedScopeError):
                with run:
                    pass

    def test_cannot_resume_with_run_id_error(self, experiment) -> None:
        """Test error is raised when both run_id and resume_last_run are set."""
        with pytest.raises(ValueError):
            Run(experiment, run_id='test-id', resume_last_run=True)

"""Test for the "experiment" module."""

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.experiment import Experiment, Run, RunIO, RunMetadata


class TestRunIO:
    """Test the RunIO class."""

    @pytest.fixture()
    def run_io(self, tmp_path) -> RunIO:
        """Set up a RunIO instance."""
        json_file = tmp_path / 'test_runs.json'
        return RunIO(json_file)

    @pytest.fixture()
    def sample_runs(self) -> list[RunMetadata]:
        """Set up sample run metadata."""
        return [
            RunMetadata(id='run1', status='completed'),
            RunMetadata(id='run2', status='failed'),
            RunMetadata(id='run3', status='running'),
        ]

    def test_init_creates_parent_directory(self, tmp_path) -> None:
        """Test that RunIO creates parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "deep" / "runs.json"
        run_io = RunIO(nested_path)
        assert nested_path.parent.exists()
        assert nested_path.exists()

    def test_init_creates_empty_json_file(self, tmp_path) -> None:
        """Test that RunIO creates an empty JSON file on initialization."""
        json_file = tmp_path / "runs.json"
        run_io = RunIO(json_file)
        assert json_file.exists()
        data = run_io.load_all()
        assert data == []

    def test_save_and_load_all(self, run_io, sample_runs) -> None:
        """Test saving and loading run metadata."""
        run_io.save_all(sample_runs)
        loaded_runs = run_io.load_all()

        assert len(loaded_runs) == 3
        assert loaded_runs[0].id == 'run1'
        assert loaded_runs[0].status == 'completed'
        assert loaded_runs[1].id == 'run2'
        assert loaded_runs[1].status == 'failed'
        assert loaded_runs[2].id == 'run3'
        assert loaded_runs[2].status == 'running'

    def test_load_all_nonexistent_file(self, tmp_path) -> None:
        """Test loading from a non-existent file returns an empty list."""
        json_file = tmp_path / "nonexistent.json"
        run_io = RunIO.__new__(RunIO)  # Create without calling __init__
        run_io.json_file = json_file

        result = run_io.load_all()
        assert result == []

    def test_load_all_corrupted_json(self, tmp_path) -> None:
        """Test loading from a corrupted JSON file returns an empty list."""
        json_file = tmp_path / "corrupted.json"
        json_file.write_text("{ invalid json }")

        run_io = RunIO.__new__(RunIO)  # Create without calling __init__
        run_io.json_file = json_file

        result = run_io.load_all()
        assert result == []

    def test_save_all_empty_list(self, run_io) -> None:
        """Test saving an empty list."""
        run_io.save_all([])
        loaded_runs = run_io.load_all()
        assert loaded_runs == []

    def test_roundtrip_data_integrity(self, run_io) -> None:
        """Test that data maintains integrity through save/load cycles."""
        original_runs = [
            RunMetadata(id='test-run-1', status='created'),
            RunMetadata(id='test-run-2', status='completed'),
        ]

        run_io.save_all(original_runs)
        loaded_runs = run_io.load_all()

        # Verify all fields match exactly
        assert len(loaded_runs) == len(original_runs)
        for original, loaded in zip(original_runs, loaded_runs, strict=False):
            assert original.id == loaded.id
            assert original.status == loaded.status


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

    def test_create_run_new(self, experiment) -> None:
        """Test creating a new run."""
        run = experiment.create_run(resume=False)
        assert isinstance(run, Run)
        assert run.experiment is experiment
        assert run.status == 'created'
        assert not run.resumed

    def test_create_run_with_custom_id(self, experiment) -> None:
        """Test creating a new run with a custom ID."""
        run = experiment.create_run(run_id='custom-id', resume=False)
        assert run.id == 'custom-id'
        assert not run.resumed

    def test_create_run_resume_no_previous_runs_error(self, experiment) -> None:
        """Test that resuming with no previous runs raises an error."""
        with pytest.raises(ValueError, match="No previous runs found"):
            experiment.create_run(resume=True)

    def test_create_run_resume_nonexistent_run_id_error(self,
                                                        experiment) -> None:
        """Test that resuming with a nonexistent run ID raises an error."""
        # Create a run first to have some data
        experiment.create_run(run_id='existing-run', resume=False)

        with pytest.raises(ValueError, match="Run nonexistent-run not found"):
            experiment.create_run(run_id='nonexistent-run', resume=True)

    def test_run_property_no_active_run_error(self, experiment) -> None:
        """Test accessing run property with no active run raises an error."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            _ = experiment.run

    def test_validate_chars_invalid_name_error(self, config, tmp_path) -> None:
        """Test invalid characters in the experiment name raise an error."""
        with pytest.raises(ValueError, match="Name contains invalid character"):
            Experiment(config, name='Invalid*Name', par_dir=tmp_path)

    def test_validate_chars_invalid_run_id_error(self, experiment) -> None:
        """Test that invalid characters in run ID raise an error."""
        with pytest.raises(ValueError, match="Name contains invalid character"):
            experiment.create_run(run_id='invalid|id', resume=False)


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
        return experiment.create_run(resume=False)

    def test_start_and_stop_run(
            self, run, experiment, config, tmp_path
    ) -> None:
        """Test starting and stopping a run using the context manager."""
        with run:
            assert run.status == 'running'
            assert Experiment.get_current() is experiment
            assert Experiment.get_current().par_dir == tmp_path
            assert Experiment.get_config() is config

        assert run.status == 'completed'
        with pytest.raises(exceptions.NoActiveExperimentError):
            Experiment.get_current()

    def test_run_is_added_to_experiment_runs_list(self, experiment) -> None:
        """Test that a new run is added to the experiment's run list."""
        experiment.previous_runs.clear()
        run1 = experiment.create_run(run_id='run1', resume=False)
        run2 = experiment.create_run(run_id='run2', resume=False)
        assert len(experiment.previous_runs) == 2
        assert experiment.previous_runs == [run1, run2]

    def test_nested_scope_error(self, run) -> None:
        """Test that an error is raised for nested runs."""
        with run:
            run2 = run.experiment.create_run(run_id='nested-run', resume=False)
            with pytest.raises(exceptions.NestedScopeError):
                with run2:
                    pass

    def test_run_status_on_exception(self, run) -> None:
        """Test that run status is set to 'failed' when an exception occurs."""
        with pytest.raises(RuntimeError):
            with run:
                raise RuntimeError("Test exception")

        assert run.status == 'failed'

    def test_run_direct_constructor(self, experiment) -> None:
        """Test creating a Run directly with the constructor."""
        run = Run(experiment, run_id='direct-run')
        assert run.id == 'direct-run'
        assert run.experiment is experiment
        assert run.status == 'created'
        assert not run.resumed
        assert experiment._active_run is run

    def test_run_constructor_resumed(self, experiment) -> None:
        """Test creating a Run with resumed=True."""
        run = Run(experiment, run_id='resumed-run', resumed=True)
        assert run.resumed
        # Resumed runs should not be added to previous_runs
        assert run not in experiment.previous_runs

    def test_run_not_resumed_added_to_previous_runs(self, experiment) -> None:
        """Test that non-resumed runs are added to previous_runs."""
        initial_count = len(experiment.previous_runs)
        run = Run(experiment, run_id='new-run', resumed=False)
        assert len(experiment.previous_runs) == initial_count + 1
        assert run in experiment.previous_runs

    def test_experiment_run_property_access(self, experiment) -> None:
        """Test accessing the current run through the experiment."""
        run = experiment.create_run(resume=False)
        assert experiment.run is run

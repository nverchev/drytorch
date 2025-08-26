"""Test for the "experiment" module."""

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.experiment import Experiment, Run, RunMetadata, RunRegistry


class TestRunRegistry:
    """Test the RunIO class."""

    @pytest.fixture()
    def registry(self, tmp_path) -> RunRegistry:
        """Set up a RunIO instance."""
        json_file = tmp_path / 'test_runs.json'
        return RunRegistry(json_file)

    @pytest.fixture()
    def sample_runs(self) -> list[RunMetadata]:
        """Set up sample run metadata."""
        return [
            RunMetadata(id='run1', status='completed', timestamp='1245'),
            RunMetadata(id='run2', status='failed', timestamp='1246'),
            RunMetadata(id='run3', status='running', timestamp='1247'),
        ]

    def test_init_creates_parent_directory(self, tmp_path) -> None:
        """Test it creates parent directories if they don't exist."""
        nested_path = tmp_path / 'nested' / 'deep' / 'runs.json'
        _ = RunRegistry(nested_path)
        assert nested_path.parent.exists()
        assert nested_path.exists()

    def test_init_creates_empty_json_file(self, tmp_path) -> None:
        """Test it creates an empty JSON file on initialization."""
        json_file = tmp_path / 'runs.json'
        run_io = RunRegistry(json_file)
        assert json_file.exists()
        data = run_io.load_all()
        assert data == []

    def test_save_and_load_all(self, registry, sample_runs) -> None:
        """Test saving and loading run metadata."""
        registry.save_all(sample_runs)
        loaded_runs = registry.load_all()

        assert len(loaded_runs) == 3
        assert loaded_runs[0].id == 'run1'
        assert loaded_runs[0].status == 'completed'
        assert loaded_runs[0].timestamp == '1245'
        assert loaded_runs[1].id == 'run2'
        assert loaded_runs[1].status == 'failed'
        assert loaded_runs[1].timestamp == '1246'
        assert loaded_runs[2].id == 'run3'
        assert loaded_runs[2].status == 'running'
        assert loaded_runs[2].timestamp == '1247'

    def test_load_all_nonexistent_file(self, tmp_path) -> None:
        """Test loading from a non-existent file returns an empty list."""
        json_file = tmp_path / 'nonexistent.json'
        run_io = RunRegistry.__new__(RunRegistry)
        run_io.file_path = json_file
        result = run_io.load_all()
        assert result == []

    def test_load_all_corrupted_json(self, tmp_path) -> None:
        """Test loading from a corrupted JSON file returns an empty list."""
        json_file = tmp_path / 'corrupted.json'
        json_file.write_text('{ invalid json }')
        run_io = RunRegistry.__new__(RunRegistry)
        run_io.file_path = json_file
        result = run_io.load_all()
        assert result == []

    def test_save_all_empty_list(self, registry) -> None:
        """Test saving an empty list."""
        registry.save_all([])
        loaded_runs = registry.load_all()
        assert loaded_runs == []

    def test_roundtrip_data_integrity(self, registry) -> None:
        """Test that data maintains integrity through save/load cycles."""
        original_runs = [
            RunMetadata(id='test-run-1', status='created', timestamp='1245'),
            RunMetadata(id='test-run-2', status='completed', timestamp='1245'),
        ]
        registry.save_all(original_runs)
        loaded_runs = registry.load_all()
        assert len(loaded_runs) == len(original_runs)
        for original, loaded in zip(original_runs, loaded_runs, strict=False):
            assert original.id == loaded.id
            assert original.status == loaded.status
            assert original.timestamp == loaded.timestamp


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
        run = experiment.create_run()
        assert isinstance(run, Run)
        assert run.experiment is experiment
        assert run.status == 'created'
        assert not run.resumed

    def test_create_run_with_custom_id(self, experiment) -> None:
        """Test creating a new run with a custom ID."""
        run = experiment.create_run(run_id='custom-id')
        assert run.id == 'custom-id'
        assert not run.resumed

    def test_create_run_resume_no_previous_runs_error(self, experiment) -> None:
        """Test that resuming with no previous runs raises an error."""
        with pytest.warns(exceptions.NoPreviousRunsWarning):
            experiment.create_run(resume=True)

    def test_create_run_resume_nonexistent_run_id_error(
        self, experiment
    ) -> None:
        """Test that resuming with a nonexistent run ID raises an error."""
        experiment.create_run(run_id='other-run')
        with pytest.warns(exceptions.NotExistingRunWarning):
            experiment.create_run(run_id='nonexistent-run', resume=True)

    def test_run_property_no_active_run_error(self, experiment) -> None:
        """Test accessing run property with no active run raises an error."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            _ = experiment.run

    def test_validate_chars_invalid_name_error(self, config, tmp_path) -> None:
        """Test invalid characters in the experiment name raise an error."""
        with pytest.raises(ValueError, match='Name contains invalid character'):
            Experiment(config, name='Invalid*Name', par_dir=tmp_path)

    def test_validate_chars_invalid_run_id_error(self, experiment) -> None:
        """Test that invalid characters in run ID raise an error."""
        with pytest.raises(ValueError, match='Name contains invalid character'):
            experiment.create_run(run_id='invalid|id', resume=False)


class TestRun:
    """Test the Run class and its context management."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up mocks for event logging."""
        self.mock_start_exp = mocker.patch.object(
            log_events, 'StartExperimentEvent'
        )
        self.mock_stop_exp = mocker.patch.object(
            log_events, 'StopExperimentEvent'
        )
        self.mock_load = mocker.patch.object(RunRegistry, 'load_all')
        self.mock_save = mocker.patch.object(RunRegistry, 'save_all')
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
        self.mock_start_exp.reset_mock()
        run.start()
        assert run.status == 'running'
        assert Experiment.get_current() is experiment
        assert Experiment.get_current().par_dir == tmp_path
        assert Experiment.get_config() is config
        assert experiment._active_run is run
        self.mock_start_exp.assert_called_once()

        run.stop()
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
                raise RuntimeError('Test exception')

        assert run.status == 'failed'

    def test_run_direct_constructor(self, experiment) -> None:
        """Test creating a Run directly with the constructor."""
        run = Run(experiment, run_id='direct-run')
        assert run.id == 'direct-run'
        assert run.experiment is experiment
        assert run.status == 'created'
        assert not run.resumed

    def test_run_constructor_resumed(self, experiment) -> None:
        """Test creating a Run with resumed=True."""
        run = Run(experiment, run_id='resumed-run', resumed=True)
        assert run.resumed
        assert run not in experiment.previous_runs

    def test_run_not_resumed_added_to_previous_runs(self, experiment) -> None:
        """Test that non-resumed runs are added to previous_runs."""
        initial_count = len(experiment.previous_runs)
        run = Run(experiment, run_id='new-run', resumed=False)
        assert len(experiment.previous_runs) == initial_count + 1
        assert run in experiment.previous_runs

    def test_is_active_status(self, run) -> None:
        """Test the is_active method returns the correct status."""
        assert not run.is_active()

        run.start()
        assert run.is_active()

        run.stop()
        assert not run.is_active()

    def test_double_start_warning(self, run) -> None:
        """Test that starting an already started run issues a warning."""
        with run:
            with pytest.warns(exceptions.RunAlreadyRunningWarning):
                run.start()

            # warn without changing status
            assert run.status == 'running'

    def test_stop_without_start_warning(self, run) -> None:
        """Test that stopping a never-started run issues a warning."""
        with pytest.warns(exceptions.RunNotStartedWarning):
            run.stop()

        # warn without changing status
        assert run.status == 'created'

    def test_double_stop_warning(self, run) -> None:
        """Test that stopping an already completed run issues a warning."""
        run.start()
        run.stop()

        with pytest.warns(exceptions.RunAlreadyCompletedWarning):
            run.stop()

        # warn without changing status
        assert run.status == 'completed'

    def test_stop_failed_run_no_warning(self, run) -> None:
        """Test stopping a failed run keep the status."""
        run.start()
        run.status = 'failed'
        run.stop()
        assert run.status == 'failed'

    def test_cleanup_resources_static_method(self, experiment, mocker) -> None:
        """Test the _cleanup_resources static method directly."""
        self.mock_stop_exp.reset_mock()
        mock_set_auto_publish = mocker.patch.object(
            log_events.Event, 'set_auto_publish'
        )
        mock_clear_current = mocker.patch.object(Experiment, '_clear_current')
        experiment._active_run = mocker.Mock()

        Run._cleanup_resources(experiment)

        assert experiment._active_run is None
        self.mock_stop_exp.assert_called_once_with(experiment.name)
        mock_set_auto_publish.assert_called_once_with(None)
        mock_clear_current.assert_called_once()

    def test_update_registry_creates_new_entry(self, run, mocker) -> None:
        """Test that _update_registry creates a new entry for a new run."""
        self.mock_load.return_value = []
        self.mock_load.reset_mock()
        self.mock_save.reset_mock()
        run._update_registry()
        self.mock_load.assert_called_once()
        self.mock_save.assert_called_once()

    def test_update_registry_updates_existing_entry(self, run, mocker) -> None:
        """Test that _update_registry updates an existing run entry."""
        existing_run_data = mocker.Mock()
        existing_run_data.id = run.id
        existing_run_data.status = 'running'
        other_run_data = mocker.Mock()
        other_run_data.id = 'other-run'
        other_run_data.status = 'failed'
        self.mock_load.return_value = [existing_run_data, other_run_data]
        self.mock_load.reset_mock()
        self.mock_save.reset_mock()
        run._update_registry()
        self.mock_load.assert_called_once()
        self.mock_save.assert_called_once()
        run.status = 'completed'
        self.mock_load.reset_mock()
        self.mock_save.reset_mock()
        run._update_registry()
        self.mock_load.assert_called_once()
        self.mock_save.assert_called_once()
        assert existing_run_data.status == 'completed'
        assert other_run_data.status == 'failed'  # unchanged
        saved_data = self.mock_save.call_args[0][0]
        assert len(saved_data) == 2
        assert existing_run_data in saved_data
        assert other_run_data in saved_data

    def test_update_registry_called_on_start_and_stop(
        self, run, mocker
    ) -> None:
        """Test _update_registry is called when starting and stopping runs."""
        mock_update = mocker.patch.object(run, '_update_registry')
        run.start()
        mock_update.assert_called()
        mock_update.reset_mock()
        run.stop()
        mock_update.assert_called()

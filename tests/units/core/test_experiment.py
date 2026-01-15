"""Test for the "experiment" module."""

from collections.abc import Generator

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.experiment import Experiment, Run, RunMetadata, RunRegistry


class _ExperimentSubclass(Experiment):
    pass


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
            RunMetadata(
                id='run1', status='completed', timestamp='1245', commit=None
            ),
            RunMetadata(
                id='run2',
                status='failed',
                timestamp='1246',
                commit='example_commit',
            ),
            RunMetadata(
                id='run3',
                status='running',
                timestamp='1247',
                commit='example_commit',
            ),
        ]

    def test_init_creates_parent_directory(self, registry) -> None:
        """Test it creates parent directories if they don't exist."""
        assert registry.file_path.parent.exists()
        assert not registry.file_path.exists()

    def test_load_all_nonexistent_file(self, registry) -> None:
        """Test loading from a non-existent file returns an empty list."""
        assert registry.load_all() == []

    def test_register_and_load_all(self, registry, sample_runs) -> None:
        """Test registering and loading run metadata."""
        for run in sample_runs:
            registry.register_new_run(run)

        loaded_runs = registry.load_all()
        assert len(loaded_runs) == 3
        assert loaded_runs[0].id == 'run1'
        assert loaded_runs[0].status == 'completed'
        assert loaded_runs[0].timestamp == '1245'
        assert loaded_runs[0].commit is None
        assert loaded_runs[1].id == 'run2'
        assert loaded_runs[1].status == 'failed'
        assert loaded_runs[1].timestamp == '1246'
        assert loaded_runs[1].commit == 'example_commit'
        assert loaded_runs[2].id == 'run3'
        assert loaded_runs[2].status == 'running'
        assert loaded_runs[2].timestamp == '1247'
        assert loaded_runs[2].commit == 'example_commit'

    def test_load_all_corrupted_json(self, tmp_path) -> None:
        """Test loading from a corrupted JSON file returns an empty list."""
        json_file = tmp_path / 'corrupted.json'
        json_file.write_text('{ invalid json }')
        run_io = RunRegistry(json_file)
        result = run_io.load_all()
        assert result == []

    def test_roundtrip_data_integrity(self, registry, sample_runs) -> None:
        """Test that data maintains integrity through save/load cycles."""
        for run in sample_runs:
            registry.register_new_run(run)

        loaded_runs = registry.load_all()
        assert len(loaded_runs) == len(sample_runs)
        for original, loaded in zip(sample_runs, loaded_runs, strict=False):
            assert original.id == loaded.id
            assert original.status == loaded.status
            assert original.timestamp == loaded.timestamp

    def test_update_status_nonexistent_run(self, registry) -> None:
        """Test error when updating a well-formatted run not in the registry."""
        with pytest.raises(exceptions.RunNotRecordedError):
            registry.update_run_status('nonexistent', 'completed')


class TestExperiment:
    """Test the Experiment class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        mocker.patch.object(log_events, 'StartExperimentEvent')
        mocker.patch.object(log_events, 'StopExperimentEvent')
        Experiment.previous_runs.clear()
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
    def started_experiment(
        self, experiment
    ) -> Generator[Experiment, None, None]:
        """Set up an experiment."""
        orig = Experiment._Experiment__current
        Experiment.__Experiment__current = experiment
        yield experiment

        Experiment.__Experiment__current = orig
        return

    @pytest.fixture
    def run1_id(self) -> str:
        """Set up a run ID fixture."""
        return 'first_run'

    @pytest.fixture
    def run2_id(self) -> str:
        """Set up a run ID fixture."""
        return 'second_run'

    @pytest.fixture
    def run1(self, experiment, run1_id) -> Run:
        """Set up a run fixture."""
        return experiment.create_run(run_id=run1_id)

    @pytest.fixture
    def run2(self, experiment, run2_id) -> Run:
        """Set up a second run fixture."""
        return experiment.create_run(run_id=run2_id)

    def test_validate_chars_invalid_name_error(self, config, tmp_path) -> None:
        """Test invalid characters in the experiment name raise an error."""
        with pytest.raises(ValueError, match='Name contains invalid character'):
            Experiment(config, name='Invalid*Name', par_dir=tmp_path)

    def test_no_active_experiment_error(self, experiment) -> None:
        """Test that an error is raised when no experiment is active."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            Experiment.get_current()

    def test_get_current_type_error(self, started_experiment) -> None:
        """Test specific error if current experiment is wrong type."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            _ExperimentSubclass.get_current()

    def test_run_property_no_active_run_error(self, experiment) -> None:
        """Test accessing run property with no active run raises an error."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            _ = experiment.run

    def test_create_run_resume_no_previous_runs_error(self, experiment) -> None:
        """Test that resuming with no previous runs raises an error."""
        with pytest.warns(exceptions.NoPreviousRunsWarning):
            experiment.create_run(resume=True)

    def test_create_run_new(self, experiment, run1, run1_id) -> None:
        """Test creating a new run."""
        assert isinstance(run1, Run)
        assert run1.id == run1_id
        assert run1.experiment is experiment
        assert run1.status == 'created'
        assert not run1.resumed

    def test_create_run_collision_error(
        self, experiment, run1, run1_id
    ) -> None:
        """Test that creating a run with an existing ID raises an error."""
        with pytest.raises(exceptions.RunAlreadyRecordedError):
            experiment.create_run(run_id=run1_id, resume=False)

    def test_create_run_resume_nonexistent_run_id_error(
        self,
        experiment,
        run1,
    ) -> None:
        """Test that resuming with a nonexistent run ID raises an error."""
        with pytest.warns(exceptions.NotExistingRunWarning):
            experiment.create_run(run_id='nonexistent-run', resume=True)

    def test_validate_chars_invalid_run_id_error(self, experiment) -> None:
        """Test that invalid characters in run ID raise an error."""
        with pytest.raises(ValueError, match='Name contains invalid character'):
            experiment.create_run(run_id='invalid|id', resume=False)

    def test_create_run_resume_from_previous_last(
        self, experiment, run1, run1_id
    ) -> None:
        """Resume the last run from memory."""
        run_resumed = experiment.create_run(resume=True)

        assert run_resumed.id == run1_id
        assert run_resumed.resumed

    def test_create_run_resume_specific_from_previous(
        self, experiment, run1, run2, run1_id
    ) -> None:
        """Resume specific run from memory."""
        r_resumed = experiment.create_run(run_id=run1_id, resume=True)

        assert r_resumed.id == run1_id
        assert r_resumed.resumed

    def test_resume_duplicate_in_memory_raises(
        self, experiment, run1, run1_id
    ) -> None:
        """Error if duplicate run IDs in memory."""
        experiment.previous_runs.append(run1)  # create a duplicate

        with pytest.raises(RuntimeError, match='Multiple runs'):
            experiment.create_run(run_id=run1_id, resume=True)

    def test_active_run_setter(self, experiment, run1) -> None:
        """Test setting active run manually."""
        experiment.run = run1
        assert experiment.run is run1

    def test_experiment_repr(self, experiment) -> None:
        """Test representation."""
        assert str(experiment) == f'Experiment(name={experiment.name})'


class TestRun:
    """Test the Run class and its context management."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up mocks for event logging."""
        self.patch_start = mocker.patch.object(
            log_events, 'StartExperimentEvent'
        )
        self.patch_stop = mocker.patch.object(log_events, 'StopExperimentEvent')
        self.patch_load = mocker.patch.object(RunRegistry, 'load_all')
        self.patch_register = mocker.patch.object(
            RunRegistry, 'register_new_run'
        )
        self.patch_update = mocker.patch.object(
            RunRegistry, 'update_run_status'
        )
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
        self.patch_start.reset_mock()
        run.start()

        assert run.status == 'running'
        assert Experiment.get_current() is experiment
        assert Experiment.get_current().par_dir == tmp_path
        assert Experiment.get_config() is config
        assert experiment._active_run is run
        self.patch_start.assert_called_once()

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
        """Test creating a Run directly with the Initialize."""
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

    def test_stop_experiment_static_method(
        self, experiment, run, mocker
    ) -> None:
        """Test the _cleanup_resources static method directly."""
        self.patch_stop.reset_mock()
        mock_set_auto_publish = mocker.patch.object(
            log_events.Event, 'set_auto_publish'
        )
        mock_clear_current = mocker.patch.object(Experiment, '_clear_current')
        experiment._active_run = mocker.Mock()
        Run._stop_experiment(experiment, run.id)

        assert experiment._active_run is None
        self.patch_stop.assert_called_once_with(experiment.name, run.id)
        mock_set_auto_publish.assert_called_once_with(None)
        mock_clear_current.assert_called_once()

    def test_update_registry_updates_existing_entry(self, run) -> None:
        """Test that _update_registry updates an existing run entry."""
        self.patch_update.reset_mock()
        run.status = 'completed'
        run._update_registry()
        self.patch_update.assert_called_once_with(run.id, 'completed')

    def test_update_registry_called_on_start_and_stop(self, run) -> None:
        """Test _update_registry is called when starting and stopping runs."""
        run.start()
        self.patch_update.assert_called()
        self.patch_update.reset_mock()
        run.stop()
        self.patch_update.assert_called()

    def test_run_repr(self, experiment) -> None:
        """Test representation."""
        run = experiment.create_run(run_id='fixed_id', resume=False)
        assert repr(run) == 'Run(id=fixed_id, status=created)'

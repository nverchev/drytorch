"""Test for the "experimenting" module."""

import warnings

from collections.abc import Generator

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.experimenting import (
    Experiment,
    Run,
    RunMetadata,
    RunRegistry,
)


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

    def test_create_run_resume_last(self, experiment, run1, run1_id) -> None:
        """Resume the last run."""
        run_resumed = experiment.create_run(resume=True)

        assert run_resumed.id == run1_id
        assert run_resumed.resumed

    def test_create_run_resume_specific(
        self, experiment, run1, run2, run1_id
    ) -> None:
        """Resume specific run."""
        r_resumed = experiment.create_run(run_id=run1_id, resume=True)

        assert r_resumed.id == run1_id
        assert r_resumed.resumed

    def test_create_run_without_record(self, experiment) -> None:
        """Test a run that is not recorded leaves the registry empty."""
        run = experiment.create_run(record=False)

        assert not run.record
        assert experiment._registry.load_all() == []

    def test_create_run_retries_taken_id(self, experiment, mocker) -> None:
        """Test a new run retries when its automatic id is already recorded."""
        error = exceptions.RunAlreadyRecordedError('taken_id', 'Experiment')
        register = mocker.patch.object(
            RunRegistry, 'register_new_run', side_effect=[error, None]
        )
        sleep = mocker.patch('time.sleep')

        experiment.create_run()

        assert register.call_count == 2
        sleep.assert_called_once_with(1)

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
        self.patch_continue = mocker.patch.object(
            log_events, 'ContinueExperimentEvent'
        )
        self.patch_stop = mocker.patch.object(log_events, 'StopExperimentEvent')
        self.patch_pause = mocker.patch.object(
            log_events, 'PauseExperimentEvent'
        )
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
    def experiment(self, config, tmp_path) -> Generator[Experiment, None, None]:
        """Set up an experiment."""
        exp = Experiment(config, name='Experiment', par_dir=tmp_path)
        try:
            yield exp
        finally:
            Run._teardown(exp)

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
        status_after_start = run.status
        current_after_start = Experiment.get_current()
        active_run_after_start = experiment._active_run
        start_called_once = self.patch_start.call_count == 1
        config_after_start = Experiment.get_config()
        run.stop()
        status_after_stop = run.status
        with pytest.raises(exceptions.NoActiveExperimentError):
            Experiment.get_current()

        assert status_after_start == 'running'
        assert current_after_start is experiment
        assert current_after_start.par_dir == tmp_path
        assert config_after_start is config
        assert active_run_after_start is run
        assert start_called_once
        assert status_after_stop == 'completed'

    def test_pause_and_resume_run(
        self, run, experiment, config, tmp_path
    ) -> None:
        """Test pausing and resuming a run."""
        self.patch_start.reset_mock()
        run.start()
        self.patch_pause.reset_mock()
        run.pause()
        status_after_pause = run.status
        with pytest.raises(exceptions.NoActiveExperimentError):
            Experiment.get_current()

        pause_called = self.patch_pause.call_count
        self.patch_start.reset_mock()
        self.patch_continue.reset_mock()
        run.start()
        status_after_continue = run.status
        current_after_continue = Experiment.get_current()
        start_called = self.patch_start.call_count
        continue_called = self.patch_continue.call_count
        run.stop()

        assert status_after_pause == 'paused'
        assert pause_called == 1
        assert status_after_continue == 'running'
        assert current_after_continue is experiment
        assert start_called == 0
        assert continue_called == 1

    @pytest.mark.parametrize(
        (
            'initial_status',
            'verb',
            'expected_status',
            'expected_warning',
            'expected_parked',
            'expected_event',
        ),
        [
            ('created', 'start', 'running', None, False, 'patch_start'),
            (
                'running',
                'start',
                'running',
                exceptions.RunAlreadyRunningWarning,
                False,
                None,
            ),
            ('paused', 'start', 'running', None, False, 'patch_continue'),
            (
                'completed',
                'start',
                'completed',
                exceptions.RunAlreadyCompletedWarning,
                False,
                None,
            ),
            (
                'failed',
                'start',
                'failed',
                exceptions.RunAlreadyFailedWarning,
                False,
                None,
            ),
            (
                'created',
                'pause',
                'created',
                exceptions.RunNotStartedWarning,
                False,
                None,
            ),
            ('running', 'pause', 'paused', None, True, 'patch_pause'),
            (
                'paused',
                'pause',
                'paused',
                exceptions.RunAlreadyPausedWarning,
                True,
                None,
            ),
            (
                'completed',
                'pause',
                'completed',
                exceptions.RunAlreadyCompletedWarning,
                False,
                None,
            ),
            (
                'failed',
                'pause',
                'failed',
                exceptions.RunAlreadyFailedWarning,
                False,
                None,
            ),
            (
                'created',
                'stop',
                'created',
                exceptions.RunNotStartedWarning,
                False,
                None,
            ),
            ('running', 'stop', 'completed', None, False, 'patch_stop'),
            ('paused', 'stop', 'completed', None, False, None),
            (
                'completed',
                'stop',
                'completed',
                exceptions.RunAlreadyCompletedWarning,
                False,
                None,
            ),
            (
                'failed',
                'stop',
                'failed',
                exceptions.RunAlreadyFailedWarning,
                False,
                None,
            ),
        ],
    )
    def test_run_state_machine(
        self,
        experiment,
        initial_status,
        verb,
        expected_status,
        expected_warning,
        expected_parked,
        expected_event,
    ) -> None:
        """Test all valid and invalid transitions in the Run state machine."""
        run = experiment.create_run(run_id='test-run', resume=False)

        if initial_status == 'running':
            run.start()
        elif initial_status == 'paused':
            run.start()
            run.pause()
        elif initial_status == 'completed':
            run.start()
            run.stop()
        elif initial_status == 'failed':
            run.start()
            run.status = 'failed'

        self.patch_update.reset_mock()
        self.patch_start.reset_mock()
        self.patch_continue.reset_mock()
        self.patch_pause.reset_mock()
        self.patch_stop.reset_mock()

        if expected_warning:
            with pytest.warns(expected_warning):
                getattr(run, verb)()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                getattr(run, verb)()

        assert run.status == expected_status
        assert (run.id in experiment.paused_runs) == expected_parked

        if expected_warning:
            self.patch_update.assert_not_called()
        else:
            self.patch_update.assert_called_once_with(run.id, expected_status)

        if expected_event:
            getattr(self, expected_event).assert_called_once()
        else:
            self.patch_start.assert_not_called()
            self.patch_continue.assert_not_called()
            self.patch_pause.assert_not_called()
            self.patch_stop.assert_not_called()

        Experiment._clear_current()

    def test_context_manager_clean_pause(self, experiment) -> None:
        """Test pausing inside the context manager keeps the run parked."""
        with experiment.create_run(run_id='clean-pause') as run:
            run.pause()

        assert run.status == 'paused'
        assert run.id in experiment.paused_runs

    def test_context_manager_raise_while_paused(self, experiment) -> None:
        """Test raising an exception while paused marks the run failed."""
        with pytest.raises(RuntimeError):
            with experiment.create_run(run_id='raise-paused') as run:
                run.pause()
                raise RuntimeError('test exception')

        assert run.status == 'failed'
        assert run.id not in experiment.paused_runs

    def test_context_manager_raise_while_running(self, experiment) -> None:
        """Test raising an exception while running marks the run as failed."""
        with pytest.raises(RuntimeError):
            with experiment.create_run(run_id='raise-running') as run:
                raise RuntimeError('test exception')

        assert run.status == 'failed'
        assert run.id not in experiment.paused_runs
        with pytest.raises(exceptions.NoActiveExperimentError):
            Experiment.get_current()

    def test_context_manager_reentry_after_pause(
        self, experiment, mocker
    ) -> None:
        """Test re-entering a paused run publishes a ContinueExperimentEvent."""
        patch_start = mocker.patch.object(log_events, 'StartExperimentEvent')
        patch_continue = mocker.patch.object(
            log_events, 'ContinueExperimentEvent'
        )

        with experiment.create_run(run_id='reentry-run') as run:
            run.pause()

        patch_start.reset_mock()
        patch_continue.reset_mock()

        with run:
            pass

        assert run.status == 'completed'
        patch_start.assert_not_called()
        patch_continue.assert_called_once()

    def test_stop_experiment_static_method(
        self, experiment, run, mocker
    ) -> None:
        """Test the _cleanup_resources static method directly."""
        self.patch_stop.reset_mock()
        mock_set_auto_publish = mocker.patch.object(
            log_events.Event, 'set_auto_publish'
        )
        mock_clear_current = mocker.patch.object(Experiment, '_clear_current')
        experiment._active_run = run
        Run._stop_experiment(experiment, run.id)

        assert experiment._active_run is None
        self.patch_stop.assert_called_once_with(experiment.name, run.id)
        mock_set_auto_publish.assert_called_once_with(None)
        mock_clear_current.assert_called_once()

    def test_stop_experiment_static_method_paused(
        self, experiment, run, mocker
    ) -> None:
        """Test the _cleanup_resources static method directly when paused."""
        self.patch_stop.reset_mock()
        mock_clean_up = mocker.patch.object(experiment.trackers, 'clean_up')
        mock_set_auto_publish = mocker.patch.object(
            log_events.Event, 'set_auto_publish'
        )
        mock_clear_current = mocker.patch.object(Experiment, '_clear_current')
        experiment._active_run = None
        Run._stop_experiment(experiment, run.id)

        assert experiment._active_run is None
        self.patch_stop.assert_not_called()
        mock_clean_up.assert_called_once()
        mock_set_auto_publish.assert_not_called()
        mock_clear_current.assert_not_called()

    def test_update_registry_updates_existing_entry(self, run) -> None:
        """Test that _update_registry updates an existing run entry."""
        self.patch_update.reset_mock()
        run.status = 'completed'
        run._update_registry()

        self.patch_update.assert_called_once_with(run.id, 'completed')

    def test_update_registry_called_on_start_and_stop(self, run) -> None:
        """Test _update_registry is called when starting and stopping runs."""
        run.start()
        called_on_start = self.patch_update.called
        self.patch_update.reset_mock()
        run.stop()
        called_on_stop = self.patch_update.called

        assert called_on_start
        assert called_on_stop

    def test_run_repr(self, experiment) -> None:
        """Test representation."""
        run = experiment.create_run(run_id='fixed_id', resume=False)

        assert repr(run) == 'Run(id=fixed_id, status=created)'

    def test_parking_and_unparking(self, experiment) -> None:
        """Test pause() parks the run and start()/stop() unpark it."""
        run = experiment.create_run()
        run.start()
        empty_after_start1 = not experiment.paused_runs
        run.pause()
        in_after_pause1 = run.id in experiment.paused_runs
        run.start()
        in_after_start2 = run.id in experiment.paused_runs
        run.pause()
        in_after_pause2 = run.id in experiment.paused_runs
        run.stop()
        in_after_stop = run.id in experiment.paused_runs

        assert empty_after_start1
        assert in_after_pause1
        assert not in_after_start2
        assert in_after_pause2
        assert not in_after_stop

    def test_resume_parked_raises_error(self, experiment) -> None:
        """Test create_run(resume=True) on parked id raises."""
        run = experiment.create_run(run_id='parked')
        run.start()
        run.pause()
        self.patch_load.return_value = [
            RunMetadata(
                id='parked', status='paused', timestamp='now', commit=None
            )
        ]
        with pytest.raises(exceptions.RunStillPausedError):
            experiment.create_run(resume=True)

        run.stop()
        run_resumed = experiment.create_run(resume=True)
        assert run_resumed.id == 'parked'

    def test_resume_abandoned_registry_entry(self, experiment, mocker) -> None:
        """Test a paused registry entry without live handle resumes normally."""
        run_data = RunMetadata(
            id='abandoned_paused',
            status='paused',
            timestamp='2021-01-01_00:00:00',
            commit=None,
        )
        self.patch_load.return_value = [run_data]
        not_in_paused_runs = 'abandoned_paused' not in experiment.paused_runs
        run_resumed = experiment.create_run(resume=True)

        assert not_in_paused_runs
        assert run_resumed.id == 'abandoned_paused'
        assert run_resumed.resumed is True

    def test_start_continue_events(self, experiment) -> None:
        """Test start vs continue events."""
        run = experiment.create_run()
        self.patch_start.reset_mock()
        run.start()
        start_call_count = self.patch_start.call_count
        start_run_id = self.patch_start.call_args.args[3]
        run.pause()
        self.patch_continue.reset_mock()
        run.start()
        continue_call_count = self.patch_continue.call_count
        continue_args = (
            self.patch_continue.call_args[0]
            if self.patch_continue.call_args
            else None
        )
        run.stop()

        assert start_call_count == 1
        assert start_run_id == run.id
        assert continue_call_count == 1
        assert continue_args == (experiment.name, run.id)

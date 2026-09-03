"""Tests for the "wandb" module."""

import copy
import importlib.util

from collections.abc import Generator

import pytest


if not importlib.util.find_spec('wandb'):
    pytest.skip('wandb not available', allow_module_level=True)

from drytorch.core import exceptions
from drytorch.trackers.wandb import Wandb, WandbWarning


class TestWandb:
    """Tests for the Wandb tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test environment."""
        self.init_mock = mocker.patch('wandb.init')
        self.finish_mock = mocker.patch('wandb.finish')
        self.mock_api = mocker.patch('wandb.Api')
        return

    @pytest.fixture
    def tracker(self) -> Wandb:
        """Set up the instance."""
        return Wandb()

    @pytest.fixture
    def tracker_started(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[Wandb, None, None]:
        """Set up started instance."""
        tracker.notify(start_experiment_mock_event)
        yield tracker

        tracker.notify(stop_experiment_mock_event)
        return

    def test_cleanup(self, tracker_started) -> None:
        """Test correct cleaning up."""
        tracker_started.clean_up()
        self.init_mock.return_value.finish.assert_called_once()
        assert tracker_started._run is None

    def test_notify_start_experiment(
        self,
        mocker,
        tracker_started,
        example_config,
        example_exp_name,
        example_run_ts,
        start_experiment_mock_event,
        example_run_id,
        example_tags,
    ) -> None:
        """Test StartExperiment notification."""
        self.init_mock.assert_called_once_with(
            id=f'{example_exp_name}_{example_run_id}',
            dir=start_experiment_mock_event.par_dir.as_posix(),
            project=example_exp_name,
            config=example_config,
            group=example_exp_name,
            settings=mocker.ANY,
            resume=None,
            tags=example_tags,
        )

    def test_notify_metrics(
        self,
        mocker,
        tracker_started,
        epoch_metrics_mock_event,
        example_named_metrics,
    ) -> None:
        """Test Metrics notification."""
        log_mock = mocker.patch.object(tracker_started.run, 'log')
        tracker_started.run.step = 1
        tracker_started.notify(epoch_metrics_mock_event)
        model_name = epoch_metrics_mock_event.model_name
        source_name = epoch_metrics_mock_event.source_name
        expected_metrics = {
            f'{model_name}/{source_name}-{name}': value
            for name, value in example_named_metrics.items()
        }
        step_dict = {f'Progress/{model_name}': epoch_metrics_mock_event.epoch}
        log_mock.assert_called_once_with(expected_metrics | step_dict)

    def test_notify_metrics_outside_scope(
        self, tracker, epoch_metrics_mock_event
    ) -> None:
        """Test Metrics notification outside scope."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            tracker.notify(epoch_metrics_mock_event)

    def test_notify_start_resume_success(
        self,
        mocker,
        tracker,
        start_experiment_mock_event,
        example_exp_name,
    ) -> None:
        """Test resuming an existing run."""
        start_experiment_mock_event.resumed = True
        mock_run = mocker.Mock()
        mock_run.id = 'existing_run_id'
        self.mock_api.return_value.runs.return_value = [mock_run]

        tracker.notify(start_experiment_mock_event)

        self.init_mock.assert_called_once()
        assert self.init_mock.call_args[1]['id'] == 'existing_run_id'
        assert self.init_mock.call_args[1]['resume'] == 'allow'

    def test_notify_start_resume_no_run(
        self,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test resuming when no previous run exists."""
        start_experiment_mock_event.resumed = True
        self.mock_api.return_value.runs.return_value = []

        with pytest.warns(WandbWarning, match='No previous runs'):
            tracker.notify(start_experiment_mock_event)

        self.init_mock.assert_called_once()
        created_id = self.init_mock.call_args[1]['id']

        assert created_id != ''
        assert self.init_mock.call_args[1]['resume'] == 'allow'

    def test_notify_start_explicit_id(
        self,
        mocker,
        example_exp_name,
        start_experiment_mock_event,
    ) -> None:
        """Test that settings.run_id takes precedence."""
        settings = mocker.Mock()
        settings.run_id = 'explicit_id'
        settings.project = None
        settings.run_group = None
        settings.entity = None
        settings.reinit = 'default'

        tracker = Wandb(settings=settings)
        tracker.notify(start_experiment_mock_event)

        self.init_mock.assert_called_once()
        assert self.init_mock.call_args[1]['id'] == 'explicit_id'

    def test_define_metric_called_once(
        self,
        tracker_started,
        epoch_metrics_mock_event,
    ) -> None:
        """Test define_metric is not called repeatedly for the same metric."""
        tracker_started.notify(epoch_metrics_mock_event)
        define_metric_mock = tracker_started.run.define_metric
        call_count_after_first = define_metric_mock.call_count

        tracker_started.notify(epoch_metrics_mock_event)

        assert define_metric_mock.call_count > 0
        assert define_metric_mock.call_count == call_count_after_first

    def test_pause_stashes_and_continue_restores(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that pause stashes state and continue restores it."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)
        tracker.notify(continue_experiment_mock_event)

    def test_pause_start_stop_continue_keeps_state(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        stop_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that concurrent runs do not corrupt stashed state."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.run_id = 'run2'

        tracker.notify(start_2)

        stop_2 = copy.copy(stop_experiment_mock_event)
        stop_2.run_id = 'run2'
        tracker.notify(stop_2)

        tracker.notify(continue_experiment_mock_event)

    def test_continue_without_stash_raises_error(
        self, tracker, continue_experiment_mock_event
    ) -> None:
        """Test that continuing without a stash raises an error."""
        with pytest.raises(exceptions.NoStashedStateError):
            tracker.notify(continue_experiment_mock_event)

    def test_close_releases_stashes(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
    ) -> None:
        """Test that close unconditionally releases stashed resources."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)
        tracker.close()

"""Tests for the "tensorboard" module."""

import copy
import importlib.util
import pathlib

import pytest

from drytorch.core import exceptions


if not importlib.util.find_spec('tensorboard'):
    pytest.skip('tensorboard not available', allow_module_level=True)


from collections.abc import Generator

from drytorch.trackers.tensorboard import TensorBoard


class TestTensorBoard:
    """Tests for the TensorBoard tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
        self.open_browser_mock = mocker.patch('webbrowser.open')
        self.mock_popen = mocker.patch('subprocess.Popen')
        self.mock_popen.return_value.poll.return_value = None
        self.summary_writer_mock = mocker.patch(
            'torch.utils.tensorboard.SummaryWriter',
        )
        return

    @pytest.fixture
    def tracker(self, tmp_path) -> TensorBoard:
        """Set up the instance."""
        return TensorBoard(par_dir=tmp_path)

    @pytest.fixture
    def tracker_started(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[TensorBoard, None, None]:
        """Set up the instance with resume."""
        tracker.notify(start_experiment_mock_event)
        yield tracker

        tracker.notify(stop_experiment_mock_event)
        tracker.close()
        return

    def test_clean_up_retains_server(self, tracker_started):
        """Test clean_up closes writer but retains server process."""
        process = tracker_started._process
        assert process is not None
        tracker_started.clean_up()
        assert tracker_started._writer is None
        assert tracker_started._process is process
        process.terminate.assert_not_called()

    def test_notify_stop_and_start_experiment(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
        example_run_id,
    ) -> None:
        """Test experiment notifications."""
        start_experiment_mock_event.config = {'simple_config': 3}
        tracker.notify(start_experiment_mock_event)
        # log_dir should be a subdirectory of tensorboard_runs_path
        called_args = self.summary_writer_mock.call_args[1]
        called_log_dir = pathlib.Path(called_args['log_dir'])
        assert called_log_dir == tracker._get_run_dir()

        writer = tracker.writer
        tracker.notify(stop_experiment_mock_event)
        writer.close.assert_called_once()
        assert tracker._writer is None

    def test_notify_metrics(
        self, tracker_started, epoch_metrics_mock_event
    ) -> None:
        """Test there is one call for each metrics and no immediate flush."""
        tracker_started.notify(epoch_metrics_mock_event)
        n_metrics = len(epoch_metrics_mock_event.metrics)
        assert tracker_started.writer.add_scalar.call_count == n_metrics
        tracker_started.writer.flush.assert_not_called()

    def test_no_logging_before_start(
        self, tracker, epoch_metrics_mock_event
    ) -> None:
        """Test no logging occurs before experiment start."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            tracker.notify(epoch_metrics_mock_event)

    def test_tensorboard_launch_fails_on_port_conflict(self, mocker, tmp_path):
        """Test error is raised if no free ports are available."""
        port_available_mock = mocker.patch.object(
            TensorBoard, '_port_available'
        )
        port_available_mock.return_value = False
        with pytest.raises(exceptions.TrackerError):
            TensorBoard._find_free_port(start=6006, max_tries=100)

    def test_pause_stashes_and_continue_restores(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that pause stashes state and continue restores it."""
        tracker.notify(start_experiment_mock_event)
        writer = tracker.writer
        tracker.notify(pause_experiment_mock_event)
        writer.flush.assert_called_once()
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

    def test_tensorboard_launch_fails_on_os_error(
        self,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test TrackerError is raised when Popen fails with OSError."""
        self.mock_popen.side_effect = PermissionError('Permission denied')
        with pytest.raises(
            exceptions.TrackerError, match='TensorBoard failed to start'
        ):
            tracker.notify(start_experiment_mock_event)

    def test_server_reused_across_runs_same_directory(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Test server process is reused across runs serving the same logdir."""
        tracker.notify(start_experiment_mock_event)
        assert self.mock_popen.call_count == 1
        initial_process = tracker._process

        tracker.notify(stop_experiment_mock_event)
        start_2 = copy.copy(start_experiment_mock_event)
        start_2.run_id = 'run2'
        tracker.notify(start_2)

        assert tracker._process is initial_process
        assert self.mock_popen.call_count == 1

    def test_server_terminated_when_logdir_changes(
        self,
        start_experiment_mock_event,
        tmp_path,
    ) -> None:
        """Test server process is terminated when logdir changes."""
        tracker = TensorBoard(par_dir=None)
        start_experiment_mock_event.par_dir = tmp_path / 'dir1'
        tracker.notify(start_experiment_mock_event)
        initial_process = tracker._process
        assert initial_process is not None

        start_diff_dir = copy.copy(start_experiment_mock_event)
        start_diff_dir.par_dir = tmp_path / 'dir2'
        start_diff_dir.run_id = 'diff_run'
        tracker.notify(start_diff_dir)

        initial_process.terminate.assert_called_once()  # type: ignore[attr-defined]
        initial_process.wait.assert_called_once()  # type: ignore[attr-defined]
        assert self.mock_popen.call_count == 2

    def test_close_terminates_server(
        self,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test close terminates running server process."""
        tracker.notify(start_experiment_mock_event)
        process = tracker._process
        assert process is not None
        tracker.close()
        process.terminate.assert_called_once()  # type: ignore[attr-defined]
        process.wait.assert_called_once()  # type: ignore[attr-defined]
        assert tracker._process is None

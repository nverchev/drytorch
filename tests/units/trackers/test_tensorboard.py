"""Tests for the ``tensorboard`` module."""

import copy
import importlib.util
import pathlib

import pytest

from drytorch.core import exceptions


if not importlib.util.find_spec('tensorboard'):
    pytest.skip('tensorboard not available', allow_module_level=True)


from collections.abc import Generator

from drytorch.trackers import tensorboard as tensorboard_module
from drytorch.trackers.tensorboard import TensorBoard


class TestTensorBoard:
    """Tests for the ``TensorBoard`` tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
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

    def test_unrecorded_run_opens_no_writer(
        self, tracker, start_experiment_mock_event, epoch_metrics_mock_event
    ) -> None:
        """Test an unrecorded run opens no writer and logs no metric."""
        start_experiment_mock_event.record = False
        tracker.notify(start_experiment_mock_event)

        tracker.notify(epoch_metrics_mock_event)

        self.summary_writer_mock.assert_not_called()

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
        tracker.notify(continue_experiment_mock_event)

        writer.flush.assert_called_once()
        assert tracker.writer is writer

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
        writer = tracker.writer
        tracker.notify(pause_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.run_id = 'run2'
        tracker.notify(start_2)

        stop_2 = copy.copy(stop_experiment_mock_event)
        stop_2.run_id = 'run2'
        tracker.notify(stop_2)

        tracker.notify(continue_experiment_mock_event)

        assert tracker.writer is writer

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
        writer = tracker.writer
        tracker.notify(pause_experiment_mock_event)

        tracker.close()

        writer.close.assert_called_once()
        assert tracker._stashed_writers == {}

    def test_command_to_read_the_results_is_logged(
        self,
        mocker,
        tracker,
        start_experiment_mock_event,
        tmp_path,
    ) -> None:
        """Test the command serving the results is logged when a run starts."""
        mocker.patch.object(TensorBoard, '_is_notebook', return_value=False)
        mock_log = mocker.patch.object(tensorboard_module._logger, 'log')

        tracker.notify(start_experiment_mock_event)

        mock_log.assert_called_once_with(
            tensorboard_module.SERVER_LOG_LEVEL,
            'TensorBoard (%s):\ntensorboard --logdir %s',
            tmp_path.name,
            (tmp_path / TensorBoard.folder_name).resolve(),
        )

    def test_notebook_shows_the_results_inline(
        self,
        mocker,
        tracker,
        start_experiment_mock_event,
        tmp_path,
    ) -> None:
        """Test a notebook shows TensorBoard inline instead of the command."""
        mocker.patch.object(TensorBoard, '_is_notebook', return_value=True)
        mock_start = mocker.patch.object(
            tensorboard_module.tb_notebook, 'start'
        )

        tracker.notify(start_experiment_mock_event)

        logdir = tmp_path / TensorBoard.folder_name
        mock_start.assert_called_once_with(f'--logdir {logdir}')

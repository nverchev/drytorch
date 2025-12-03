"""Tests for the "tensorboard" module."""

import importlib.util
import pathlib

import pytest


if not importlib.util.find_spec('tensorboard'):
    pytest.skip('tensorboard not available', allow_module_level=True)


from collections.abc import Generator

from drytorch.core import exceptions
from drytorch.trackers.tensorboard import TensorBoard


class TestTensorBoard:
    """Tests for the TensorBoard tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
        self.open_browser_mock = mocker.patch('webbrowser.open')
        self.mock_popen = mocker.patch('subprocess.Popen')
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
        return

    def test_cleanup(self, tracker_started):
        """Test correct cleaning up."""
        tracker_started.clean_up()
        assert tracker_started._writer is None

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
        """Test there is one call for each metrics."""
        tracker_started.notify(epoch_metrics_mock_event)
        n_metrics = len(epoch_metrics_mock_event.metrics)
        assert tracker_started.writer.add_scalar.call_count == n_metrics

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

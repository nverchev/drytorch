"""Tests for the "tensorboard" module."""

import importlib.util
import pathlib

import pytest

if not importlib.util.find_spec('tensorboard'):
    pytest.skip('tensorboard not available', allow_module_level=True)

import time

from collections.abc import Generator

from drytorch import exceptions
from drytorch.trackers.tensorboard import TensorBoard


class TestTensorBoard:
    """Tests for the TensorBoard tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
        self.open_browser_mock = mocker.patch("webbrowser.open")
        self.mock_popen = mocker.patch("subprocess.Popen")
        self.summary_writer_mock = mocker.patch(
            'torch.utils.tensorboard.SummaryWriter',
        )
        return

    @pytest.fixture
    def tracker(self, tmp_path) -> TensorBoard:
        """Set up the instance."""
        return TensorBoard(par_dir=tmp_path, open_browser=True)

    @pytest.fixture
    def tracker_with_resume(self, tmp_path) -> TensorBoard:
        """Set up the instance with resume."""
        return TensorBoard(resume_run=True, open_browser=True)

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
            example_variation,
    ) -> None:
        """Test experiment notifications."""
        start_experiment_mock_event.config = {'simple_config': 3}
        tracker.notify(start_experiment_mock_event)
        if example_variation:
            par_dir = tracker.par_dir.parent
        else:
            par_dir = tracker.par_dir

        tensorboard_runs_path = par_dir / TensorBoard.folder_name

        # log_dir should be a subdirectory of tensorboard_runs_path
        called_args = self.summary_writer_mock.call_args[1]
        called_log_dir = pathlib.Path(called_args['log_dir'])
        assert called_log_dir.parent == tensorboard_runs_path

        writer = tracker.writer
        tracker.notify(stop_experiment_mock_event)
        writer.close.assert_called_once()
        assert tracker._writer is None

    def test_resume(
            self,
            mocker,
            tmp_path,
            tracker_with_resume,
            example_variation,
            start_experiment_mock_event,
            stop_experiment_mock_event,
    ) -> None:
        """Test resume previous run."""
        start_experiment_mock_event.config = {'simple_config': 3}
        last_run = mocker.patch.object(tracker_with_resume, '_get_last_run')
        last_run.return_value = tmp_path

        tracker_with_resume.notify(start_experiment_mock_event)

        called_args = self.summary_writer_mock.call_args[1]
        called_log_dir = pathlib.Path(called_args['log_dir'])
        assert called_log_dir == tmp_path

        self.summary_writer_mock.reset_mock()
        tracker_with_resume.notify(stop_experiment_mock_event)

        # no previous run -> should create a subfolder under exp_dir/folder_name
        last_run.return_value = None
        tracker_with_resume.notify(start_experiment_mock_event)

        par_dir = start_experiment_mock_event.exp_dir
        expected_parent = par_dir / TensorBoard.folder_name
        called_args = self.summary_writer_mock.call_args[1]
        called_log_dir = pathlib.Path(called_args['log_dir'])

        # assert it's a new subdirectory under the expected base directory

        assert called_log_dir.parent == expected_parent

    def test_notify_metrics(self,
                            tracker_started,
                            epoch_metrics_mock_event) -> None:
        """Test there is one call for each metrics."""
        tracker_started.notify(epoch_metrics_mock_event)
        n_metrics = len(epoch_metrics_mock_event.metrics)
        assert tracker_started.writer.add_scalar.call_count == n_metrics

    def test_no_logging_before_start(self,
                                     tracker,
                                     epoch_metrics_mock_event) -> None:
        """Test no logging occurs before experiment start."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            tracker.notify(epoch_metrics_mock_event)

    def test_get_last_run(self, tmp_path) -> None:
        """Test last created folder is selected."""
        with pytest.warns(exceptions.DryTorchWarning):
            TensorBoard._get_last_run(tmp_path)
        for i in range(3, 0, -1):
            folder_name = str(i)
            path = tmp_path / folder_name
            path.mkdir()
            time.sleep(0.01)

        assert TensorBoard._get_last_run(tmp_path) == tmp_path / '1'

    def test_tensorboard_launch_fails_on_port_conflict(self, mocker, tmp_path):
        """Test error is raised if no free ports are available."""
        port_available_mock = mocker.patch.object(TensorBoard,
                                                  '_port_available')
        port_available_mock.return_value = False
        with pytest.raises(exceptions.TrackerError):
            TensorBoard._find_free_port(start=6006, max_tries=100)

    def test_browser_open_failure_warning(self, tracker, mocker, tmp_path):
        """Test warning is issued when the browser fails to open."""
        mocker.patch.object(TensorBoard, '_find_free_port', return_value=6007)
        self.open_browser_mock.side_effect = OSError('Browser not available')
        with pytest.warns(exceptions.DryTorchWarning):
            tracker._start_tensorboard(tmp_path)

    def test_tensorboard_opens_browser_once_per_tracker(self, mocker, tmp_path):
        """Test each tracker gets a unique port and browser launch."""
        # mock _find_free_port to return different ports for each tracker
        find_port_mock = mocker.patch.object(TensorBoard, '_find_free_port')
        find_port_mock.side_effect = [6007,
                                      6008]  # different ports for each call
        tracker1 = TensorBoard(par_dir=tmp_path / 'exp1', open_browser=True)
        tracker2 = TensorBoard(par_dir=tmp_path / 'exp2', open_browser=True)
        tracker1._start_tensorboard(tmp_path / 'exp1')
        tracker2._start_tensorboard(tmp_path / 'exp2')

        assert self.open_browser_mock.call_count == 2
        self.open_browser_mock.assert_any_call('http://localhost:6007')
        self.open_browser_mock.assert_any_call('http://localhost:6008')
        assert tracker1._port == 6007
        assert tracker2._port == 6008
        assert self.mock_popen.call_count == 2

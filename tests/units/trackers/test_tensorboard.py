"""Tests for the "tensorboard" module."""

import pytest

from typing import Generator
import time

from dry_torch import exceptions
from dry_torch.trackers.tensorboard import TensorBoard


class TestTensorBoard:
    """Tests for the TensorBoard tracker."""

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
    def tracker_with_resume(self, tmp_path) -> TensorBoard:
        """Set up the instance with resume."""
        return TensorBoard(resume_run=True)

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
        tracker_started.clean_up()
        assert tracker_started._writer is None

    def test_notify_stop_and_start_experiment(
            self,
            tracker,
            start_experiment_mock_event,
            stop_experiment_mock_event,
    ) -> None:
        """Test experiment notifications."""
        start_experiment_mock_event.config = {'simple_config': 3}
        tracker.notify(start_experiment_mock_event)
        log_dir = tracker.par_dir / TensorBoard.folder_name
        self.summary_writer_mock.assert_called_once_with(
            log_dir=log_dir.as_posix()
        )
        writer = tracker.writer
        writer.add_hparams.assert_called_once_with(
            hparam_dict=start_experiment_mock_event.config,
            metric_dict={}
        )
        tracker.notify(stop_experiment_mock_event)
        writer.close.assert_called_once()
        assert tracker._writer is None

    def test_resume(self,
                    mocker,
                    tmp_path,
                    tracker_with_resume,
                    start_experiment_mock_event,
                    stop_experiment_mock_event) -> None:
        """Test resume previous run."""
        start_experiment_mock_event.config = {'simple_config': 3}
        last_run = mocker.patch.object(tracker_with_resume, '_get_last_run')
        last_run.return_value = tmp_path
        tracker_with_resume.notify(start_experiment_mock_event)
        log_dir = start_experiment_mock_event.exp_dir / TensorBoard.folder_name
        self.summary_writer_mock.assert_called_once_with(
            log_dir=tmp_path.as_posix()
        )
        self.summary_writer_mock.reset_mock()
        tracker_with_resume.notify(stop_experiment_mock_event)
        # mock case not previous experiment is retrieved
        last_run.return_value = None
        with pytest.warns(exceptions.DryTorchWarning):
            tracker_with_resume.notify(start_experiment_mock_event)
        self.summary_writer_mock.assert_called_once_with(
            log_dir=log_dir.as_posix()
        )

    def test_notify_metrics(self,
                            tracker_started,
                            epoch_metrics_mock_event) -> None:
        """Test there is one call for each metrics"""
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
        assert not TensorBoard._get_last_run(tmp_path)
        for i in range(3, 0, -1):
            folder_name = str(i)
            path = tmp_path / folder_name
            path.mkdir()
            time.sleep(0.01)

        assert TensorBoard._get_last_run(tmp_path) == tmp_path / '1'

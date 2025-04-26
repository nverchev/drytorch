"""Test TensorBoard tracker"""

import pytest
import torch.utils.tensorboard

from dry_torch import log_events
from dry_torch.trackers import tensorboard


class TestTensorBoard:
    """Tests for the TensorBoard tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, start_experiment_event):
        """Setup test environment."""
        self.summary_writer_mock = mocker.patch(
            'torch.utils.tensorboard.SummaryWriter')
        self.add_hparams_mock = mocker.patch.object(
            torch.utils.tensorboard.SummaryWriter,
            'add_hparams'
        )
        self.add_scalar_mock = mocker.patch.object(
            torch.utils.tensorboard.SummaryWriter,
            'add_scalar'
        )
        self.add_image_mock = mocker.patch.object(
            torch.utils.tensorboard.SummaryWriter,
            'add_image'
        )
        self.close_mock = mocker.patch.object(
            torch.utils.tensorboard.SummaryWriter,
            'close'
        )

        self.tracker = tensorboard.TensorBoard()
        self.tracker.notify(start_experiment_event)

    def test_notify_start_experiment(self, start_experiment_event):
        """Test StartExperiment notification."""
        # Verify SummaryWriter was initialized with correct directory
        self.summary_writer_mock.assert_called_once_with(
            log_dir=start_experiment_event.exp_dir
        )

        # Verify hyperparameters were logged
        self.add_hparams_mock.assert_called_once_with(
            hparam_dict=start_experiment_event.config,
            metric_dict={}
        )

    def test_notify_stop_experiment(self, stop_experiment_event):
        """Test StopExperiment notification."""
        self.tracker.notify(stop_experiment_event)
        self.close_mock.assert_called_once()

    def test_notify_metrics(self, epoch_metrics_event, sample_metrics):
        """Test Metrics notification."""
        self.tracker.notify(epoch_metrics_event)

        # Verify metrics were logged with correct naming
        expected_calls = [
            mocker.call(
                f'{epoch_metrics_event.model_name}/{epoch_metrics_event.source}-{name}',
                value,
                global_step=epoch_metrics_event.epoch
            )
            for name, value in sample_metrics.items()
        ]

        # Check that add_scalar was called for each metric with correct parameters
        self.add_scalar_mock.assert_has_calls(expected_calls, any_order=True)

    def test_notify_images(self, mocker, epoch_images_event, sample_images):
        """Test Images notification."""
        self.tracker.notify(epoch_images_event)

        # Verify images were logged with correct naming
        expected_calls = [
            mocker.call(
                f'{epoch_images_event.model}/{epoch_images_event.source}-{name}',
                img,
                global_step=epoch_images_event.epoch
            )
            for name, img in sample_images.items()
        ]

        # Check that add_image was called for each image with correct parameters
        self.add_image_mock.assert_has_calls(expected_calls, any_order=True)

    def test_custom_log_dir(self, mocker, start_experiment_event):
        """Test custom log directory specification."""
        custom_log_dir = "/custom/log/path"
        tracker = tensorboard.TensorBoard(log_dir=custom_log_dir)

        # Simulate start experiment
        tracker.notify(start_experiment_event)

        # Verify SummaryWriter was initialized with custom directory
        self.summary_writer_mock.assert_called_once_with(
            log_dir=custom_log_dir
        )

    def test_no_logging_before_start(self, epoch_metrics_event):
        """Test that no logging occurs before experiment start."""
        # Create a new tracker without starting an experiment
        tracker = tensorboard.TensorBoard()

        # Attempt to log metrics
        tracker.notify(epoch_metrics_event)

        # Verify no logging methods were called
        self.add_scalar_mock.assert_not_called()
        self.add_image_mock.assert_not_called()
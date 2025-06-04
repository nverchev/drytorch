"""Tests for the "wandb" module."""

import pytest
from wandb.sdk import wandb_settings

from dry_torch import log_events
from dry_torch.trackers import wandb


class TestWandb:
    """Tests for the Wandb tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, start_experiment_event):
        """Setup test environment."""
        self.init_mock = mocker.patch('wandb.init')
        self.log_mock = mocker.patch('wandb.log')
        self.finish_mock = mocker.patch('wandb.finish')

        self.tracker = wandb.Wandb()
        self.tracker.notify(start_experiment_event)

    def test_notify_start_experiment(self, start_experiment_event):
        """Test StartExperiment notification."""
        self.init_mock.assert_called_once_with(
            dir=start_experiment_event.exp_dir,
            project=start_experiment_event.exp_name,
            config=start_experiment_event.config,
            settings=self.tracker._settings
        )

    def test_notify_stop_experiment(self, stop_experiment_event):
        """Test StopExperiment notification."""
        self.tracker.notify(stop_experiment_event)
        self.finish_mock.assert_called_once()

    def test_notify_metrics(self, epoch_metrics_event, sample_metrics):
        """Test Metrics notification."""
        self.tracker.notify(epoch_metrics_event)

        expected_metrics = {
            f'{epoch_metrics_event.model_name}/{epoch_metrics_event.source}-{name}': value
            for name, value in sample_metrics.items()
        }

        self.log_mock.assert_called_once_with(
            **expected_metrics,
            step=epoch_metrics_event.epoch
        )

    def test_settings_priority(self, mocker, start_experiment_event):
        """Test settings priority."""
        init_mock = mocker.patch('wandb.init')

        custom_settings = wandb_settings.Settings(project="custom_project")
        tracker = wandb.Wandb(settings=custom_settings)

        tracker.notify(start_experiment_event)

        init_mock.assert_called_once_with(
            dir=start_experiment_event.exp_dir,
            project=None,
            config=start_experiment_event.config,
            settings=custom_settings
        )

    def test_settings_fallback(self, mocker):
        """Test settings fallback to experiment parameters."""
        init_mock = mocker.patch('wandb.init')

        custom_settings = wandb_settings.Settings(project=None)
        tracker = wandb.Wandb(settings=custom_settings)

        start_event = log_events.StartExperiment(
            exp_name="my_experiment",
            exp_dir="/path/to/exp",
            config={"param": "value"}
        )

        tracker.notify(start_event)

        init_mock.assert_called_once_with(
            dir=start_event.exp_dir,
            project="my_experiment",
            config=start_event.config,
            settings=custom_settings
        )


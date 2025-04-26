"""Functional tests for wandb tracker.

These tests are meant to be run manually to verify integration with the WandB website.
They are skipped during automated testing.
"""

import pytest
from wandb.sdk import wandb_settings

from dry_torch.trackers import wandb


class TestWandbFunctional:
    """Functional tests for WandB tracker that create actual WandB entries.

    Skip these tests during automated runs. Run them manually when you need
    to verify the behavior on the WandB website.
    """

    @pytest.fixture(autouse=True)
    def setup(self, start_experiment_event, stop_experiment_event):
        """Setup unique experiment name for every test run."""

        # Create a tracker with a unique project name
        self.tracker = wandb.Wandb(
            par_dir='.',
            settings=wandb_settings.Settings(anonymous='allow')
        )

        # Start the experiment
        self.tracker.notify(start_experiment_event)

        yield

        # Clean up
        self.tracker.notify(stop_experiment_event)


    def test_basic_metrics_logging(self, epoch_metrics_event):
        """Test basic metrics logging."""
        self.tracker.notify(epoch_metrics_event)
        epoch_metrics_event.epoch = 11
        self.tracker.notify(epoch_metrics_event)
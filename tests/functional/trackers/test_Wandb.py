"""Functional tests for Wandb tracker."""

import pytest

from typing import Generator

from wandb.sdk import wandb_settings

from dry_torch.trackers.wandb import Wandb


class TestWandbFullCycle:
    """Complete Wandb session and tests it afterward."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, event_workflow) -> None:
        """Setup unique experiment name for every test run."""
        self.settings = wandb_settings.Settings(anonymous='allow',
                                                mode='offline',
                                                root_dir=tmp_path.as_posix())
        tracker = Wandb(settings=self.settings)
        for event in event_workflow:
            tracker.notify(event)

    @pytest.fixture
    def resumed_tracker(
            self,
            start_training_event,
            stop_experiment_event,
    ) -> Generator[Wandb, None, None]:
        """Set up resumed instance."""
        tracker = Wandb(settings=self.settings, resume_run=True)
        tracker.notify(start_training_event)
        yield tracker

        tracker.notify(stop_experiment_event)
        return

    def test_local_creation(self, tmp_path):
        assert list(tmp_path.iterdir())

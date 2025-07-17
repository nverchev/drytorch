"""Functional tests for Wandb tracker."""

import pytest

try:
    import wandb
except ImportError:
    pytest.skip('wandb not available', allow_module_level=True)

from typing import Generator

from wandb.sdk import wandb_settings

from drytorch.trackers.wandb import Wandb


class TestWandbFullCycle:
    """Complete Wandb session and tests it afterward."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, event_workflow) -> None:
        """Set up a unique experiment name for every test run."""
        self.settings = wandb_settings.Settings(anonymous='allow',
                                                mode='offline',
                                                root_dir=tmp_path.as_posix())
        tracker = Wandb(settings=self.settings)
        for event in event_workflow:
            tracker.notify(event)

    @pytest.fixture
    def resumed_tracker(
            self,
            start_experiment_event,
            stop_experiment_event,
    ) -> Generator[Wandb, None, None]:
        """Set up a resumed instance."""
        tracker = Wandb(settings=self.settings, resume_run=True)
        tracker.notify(start_experiment_event)
        yield tracker

        tracker.notify(stop_experiment_event)
        return

    def test_folder_creation(self, tmp_path, example_exp_name):
        """Test that wandb creates local files and directories."""
        created_items = list((tmp_path / example_exp_name).iterdir())
        assert created_items  # Should not be empty

        # Check for wandb directory structure
        wandb_dirs = [item for item in created_items if
                      item.name.startswith('wandb')]
        assert wandb_dirs

    @pytest.mark.skip(reason='wandb does not support resuming offline runs')
    def test_resume_functionality(self,
                                  resumed_tracker,
                                  start_training_event,
                                  example_model_name,
                                  example_source_name,
                                  example_loss_name):
        """Test that resume functionality works correctly."""
        key = f'{example_model_name}/{example_source_name}-{example_loss_name}'
        summary = resumed_tracker.run.summary
        # note summary only gets the last value
        assert key in summary.keys()

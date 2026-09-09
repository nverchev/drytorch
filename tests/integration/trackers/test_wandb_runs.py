"""Tests for Wandb interleaved pause and continue."""

import pathlib

import pytest


try:
    from wandb.sdk import wandb_settings
except ImportError:
    pytest.skip('wandb not available', allow_module_level=True)
    raise

from drytorch.core import log_events
from drytorch.trackers.wandb import Wandb


class TestWandbPauseContinue:
    """Tests Wandb run restoration across interleaved runs."""

    @pytest.fixture
    def tracker(self, tmp_path: pathlib.Path) -> Wandb:
        """Create an offline Wandb tracker instance."""
        settings = wandb_settings.Settings(
            mode='offline',
            root_dir=tmp_path.as_posix(),
            anonymous='allow',
            silent=True,
        )
        return Wandb(par_dir=tmp_path, settings=settings)

    def test_interleaved_pause_continue(
        self,
        tracker: Wandb,
        start_experiment_event: log_events.StartExperimentEvent,
        pause_experiment_event: log_events.PauseExperimentEvent,
        continue_experiment_event: log_events.ContinueExperimentEvent,
        stop_experiment_event: log_events.StopExperimentEvent,
        start_experiment_event_b: log_events.StartExperimentEvent,
        stop_experiment_event_b: log_events.StopExperimentEvent,
        metrics_event_a1: log_events.MetricEvent,
        metrics_event_b: log_events.MetricEvent,
        metrics_event_a2: log_events.MetricEvent,
        example_model_name: str,
        example_source_name: str,
    ) -> None:
        """Verify Wandb run restoration and metric persistence."""
        # Trigger
        tracker.notify(start_experiment_event)
        run_a_instance = tracker.run
        tracker.notify(metrics_event_a1)
        tracker.notify(pause_experiment_event)

        tracker.notify(start_experiment_event_b)
        run_b_instance = tracker.run
        tracker.notify(metrics_event_b)
        tracker.notify(stop_experiment_event_b)

        tracker.notify(continue_experiment_event)
        run_a_resumed_instance = tracker.run
        defined_metrics_after_continue = set(tracker._defined_metrics)
        tracker.notify(metrics_event_a2)
        tracker.notify(stop_experiment_event)
        tracker.close()

        # Assert
        metric_key = f'{example_model_name}/{example_source_name}-loss'
        assert run_a_resumed_instance is run_a_instance
        assert run_b_instance is not run_a_instance
        assert run_b_instance.id != run_a_instance.id
        assert metric_key in defined_metrics_after_continue

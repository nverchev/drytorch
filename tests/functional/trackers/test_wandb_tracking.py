"""Functional tests for Wandb tracker."""

import os
import pathlib
import webbrowser

from collections.abc import Generator, Iterable

import pytest


try:
    from wandb.sdk import wandb_settings
except ImportError:
    pytest.skip('wandb not available', allow_module_level=True)
    raise

from drytorch.core import log_events
from drytorch.trackers.wandb import Wandb


@pytest.mark.filterwarnings('ignore::DeprecationWarning:wandb.*')
class TestWandbFullCycle:
    """Complete Wandb session and tests it afterward."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        tmp_path: pathlib.Path,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """Set up a unique experiment name and execute the workflow."""
        self.settings = self._create_settings(tmp_path)
        tracker = Wandb(settings=self.settings)
        url = self._run_workflow(tracker, plotting_workflow)

        if 'LIVE_PLOT' in os.environ and url:
            self._open_dashboard(url)

        return

    @staticmethod
    def _create_settings(tmp_path: pathlib.Path) -> wandb_settings.Settings:
        mode = 'online' if 'LIVE_PLOT' in os.environ else 'offline'
        return wandb_settings.Settings(
            anonymous='allow',
            mode=mode,
            root_dir=tmp_path.as_posix(),
        )

    @staticmethod
    def _run_workflow(
        tracker: Wandb,
        workflow: Iterable[log_events.Event],
    ) -> str | None:
        url = None
        for event in workflow:
            tracker.notify(event)
            if url is None and tracker._run is not None:
                url = tracker._run.url

        return url

    @staticmethod
    def _open_dashboard(url: str) -> None:
        # Silence GTK accessibility warning when launching browser
        os.environ['NO_AT_BRIDGE'] = '1'
        if 'GTK_MODULES' in os.environ:
            _modules = [
                m
                for m in os.environ['GTK_MODULES'].split(':')
                if m != 'atk-bridge'
            ]
            if _modules:
                os.environ['GTK_MODULES'] = ':'.join(_modules)
            else:
                del os.environ['GTK_MODULES']

        webbrowser.open(url)

    @pytest.fixture
    def resumed_tracker(
        self,
        start_experiment_event,
        stop_experiment_event,
    ) -> Generator[Wandb, None, None]:
        """Set up a resumed instance."""
        tracker = Wandb(settings=self.settings)
        tracker.notify(start_experiment_event)
        yield tracker

        tracker.notify(stop_experiment_event)
        return

    def test_folder_creation(self, tmp_path, example_exp_name):
        """Test that wandb creates local files and directories."""
        created_items = list((tmp_path / Wandb.folder_name).iterdir())
        assert created_items

    @pytest.mark.skip(reason='wandb does not support resuming offline runs')
    def test_resume_functionality(
        self,
        resumed_tracker,
        example_model_name,
        example_source_name,
        example_loss_name,
    ) -> None:
        """Test that resume functionality works correctly."""
        key = f'{example_model_name}/{example_source_name}-{example_loss_name}'
        summary = resumed_tracker.run.summary
        # note summary only gets the last value
        assert key in summary.keys()

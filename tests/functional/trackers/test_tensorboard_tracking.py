"""Functional tests for tracking with TensorBoard."""

import dataclasses
import importlib.util
import json
import os
import time
import urllib.request
import webbrowser

from collections.abc import Generator, Iterable

import pytest


if not importlib.util.find_spec('tensorboard'):
    pytest.skip('tensorboard not available', allow_module_level=True)

from drytorch.core import log_events
from drytorch.trackers.tensorboard import TensorBoard


@pytest.mark.filterwarnings('ignore::UserWarning:tensorboard.*')
class TestTensorBoardFullCycle:
    """Complete TensorBoard session and tests it afterward."""

    @pytest.fixture
    def setup(
        self,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> Generator[None, None, None]:
        """Set up TensorBoard tracker and execute the workflow."""
        start_server = 'LIVE_PLOT' in os.environ
        self.tracker = TensorBoard(start_server=start_server)
        self._run_workflow(self.tracker, plotting_workflow)
        yield
        self.tracker.close()

    @staticmethod
    def _run_workflow(
        tracker: TensorBoard,
        workflow: Iterable[log_events.Event],
    ) -> str | None:
        is_live = 'LIVE_PLOT' in os.environ
        default_pause = 10.0 if is_live else 0.0
        pause_final = float(os.environ.get('LIVE_PLOT_PAUSE', default_pause))

        url = None
        events = list(workflow)
        for event in events:
            if isinstance(event, log_events.StopExperimentEvent):
                break

            tracker.notify(event)
            if url is None and tracker._port is not None:
                url = f'http://localhost:{tracker._port}'

        if tracker._writer is not None:
            tracker.writer.flush()

        if is_live and url:
            TestTensorBoardFullCycle._wait_for_scalars(url)
            TestTensorBoardFullCycle._open_dashboard(url)
            time.sleep(pause_final)

        if events and isinstance(events[-1], log_events.StopExperimentEvent):
            tracker.notify(events[-1])

        return url

    @staticmethod
    def _wait_for_scalars(url: str, timeout: float = 5.0) -> None:
        """Wait until TensorBoard backend has indexed scalar metrics."""
        deadline = time.time() + timeout
        endpoint = f'{url.rstrip("/")}/data/plugin/scalars/tags'
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(  # noqa: S310
                    endpoint, timeout=1.0
                ) as resp:
                    data = json.loads(resp.read().decode())
                    if data:
                        return
            except (OSError, ValueError):
                pass

            time.sleep(0.2)

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
    ) -> Generator[TensorBoard, None, None]:
        """Set up the resumed instance."""
        tracker: TensorBoard = TensorBoard(start_server=False)
        resumed_event = dataclasses.replace(
            start_experiment_event, resumed=True
        )
        tracker.notify(resumed_event)
        yield tracker

        tracker.notify(stop_experiment_event)
        tracker.close()
        return

    def test_folder_creation(self, setup, tmp_path, example_exp_name) -> None:
        """Test that TensorBoard creates local files and logs."""
        tb_dir = tmp_path / TensorBoard.folder_name / example_exp_name
        assert tb_dir.exists()
        assert tb_dir.is_dir()

        created_files = list(tb_dir.rglob('events.out.tfevents*'))
        assert created_files
        for file in created_files:
            assert file.stat().st_size > 0

    def test_resume_functionality(self, resumed_tracker) -> None:
        """Test that resume functionality works correctly."""
        tensorboard_dir = resumed_tracker._get_run_dir()
        assert tensorboard_dir.exists()
        assert tensorboard_dir.is_dir()

        created_folders = list(tensorboard_dir.iterdir())
        assert created_folders

        for folder in created_folders:
            assert folder.name.startswith('events.out.tfevents')
            assert folder.stat().st_size > 0

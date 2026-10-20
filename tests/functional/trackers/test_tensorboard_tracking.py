"""Functional tests for tracking with TensorBoard."""

import dataclasses
import importlib.util

from collections.abc import Generator

import pytest


if not importlib.util.find_spec('tensorboard'):
    pytest.skip('tensorboard not available', allow_module_level=True)

from drytorch.core import log_events
from drytorch.trackers.tensorboard import TensorBoard


@pytest.mark.filterwarnings('ignore::UserWarning:tensorboard.*')
class TestTensorBoardFullCycle:
    """Complete ``TensorBoard`` session and tests it afterward."""

    @pytest.fixture
    def setup(
        self,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> Generator[None, None, None]:
        """Set up TensorBoard tracker and execute the workflow."""
        self.tracker = TensorBoard()
        for event in plotting_workflow:
            self.tracker.notify(event)

        yield
        self.tracker.close()

    @pytest.fixture
    def resumed_tracker(
        self,
        start_experiment_event,
        stop_experiment_event,
    ) -> Generator[TensorBoard, None, None]:
        """Set up the resumed instance."""
        tracker: TensorBoard = TensorBoard()
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

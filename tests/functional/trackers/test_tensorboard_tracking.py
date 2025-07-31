"""Functional tests for trakcing with TensorBoard."""

import importlib.util

from collections.abc import Generator

import pytest


if not importlib.util.find_spec('tensorboard'):
    pytest.skip('tensorboard not available', allow_module_level=True)
from drytorch.trackers.tensorboard import TensorBoard


class TestTensorBoardFullCycle:
    """Complete TensorBoard session and tests it afterward."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, event_workflow) -> None:
        """Set up TensorBoard tracker and run complete workflow."""
        self.tracker = TensorBoard(par_dir=tmp_path)
        for event in event_workflow:
            self.tracker.notify(event)

    @pytest.fixture
    def resumed_tracker(
        self,
        tmp_path,
        start_experiment_event,
        stop_experiment_event,
    ) -> Generator[TensorBoard, None, None]:
        """Set up the resumed instance."""
        tracker = TensorBoard(par_dir=tmp_path, resume_run=True)
        tracker.notify(start_experiment_event)
        yield tracker

        tracker.notify(stop_experiment_event)
        return

    def test_folder_creation(self,
                             resumed_tracker,
                             tmp_path,
                             example_exp_name):
        """Test that TensorBoard creates local files and logs."""
        tensorboard_dir = tmp_path / resumed_tracker.folder_name
        assert tensorboard_dir.exists()
        assert tensorboard_dir.is_dir()

        created_folders = list(tensorboard_dir.iterdir())
        assert created_folders

        for folder in created_folders:
            assert folder.name.startswith(example_exp_name)
            assert folder.stat().st_size > 0

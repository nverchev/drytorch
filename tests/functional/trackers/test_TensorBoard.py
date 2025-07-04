"""Functional tests for TensorBoard tracker."""

import pytest

try:
    import torch.utils.tensorboard
except ImportError:
    pytest.skip('tensorboard not available', allow_module_level=True)

from typing import Generator

from drytorch.trackers.tensorboard import TensorBoard


class TestTensorBoardFullCycle:
    """Complete TensorBoard session and tests it afterward."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, event_workflow) -> None:
        """Setup TensorBoard tracker and run complete workflow."""
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
        """Set up resumed instance."""
        tracker = TensorBoard(par_dir=tmp_path, resume_run=True)
        tracker.notify(start_experiment_event)
        yield tracker

        tracker.notify(stop_experiment_event)
        return

    def test_folder_creation(self, tmp_path, example_named_metrics):
        """Test that TensorBoard creates local files and logs."""
        tensorboard_dir = tmp_path / TensorBoard.folder_name
        assert tensorboard_dir.exists()
        assert tensorboard_dir.is_dir()

        created_items = list(tensorboard_dir.iterdir())
        assert created_items

        event_files = [item for item in created_items if
                       item.name.startswith('events.out.tfevents')]
        assert event_files
        for event_file in event_files:
            assert event_file.stat().st_size > 0

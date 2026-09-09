"""Tests for TensorBoard interleaved pause and continue."""

import pathlib

from collections.abc import Callable

import pytest


try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    pytest.skip('tensorboard not available', allow_module_level=True)
    raise

from drytorch.core import log_events
from drytorch.trackers.tensorboard import TensorBoard


class TestTensorBoardPauseContinue:
    """Tests TensorBoard state isolation across interleaved runs."""

    @pytest.fixture
    def tracker(self, tmp_path: pathlib.Path) -> TensorBoard:
        """Create a TensorBoard instance with server disabled."""
        return TensorBoard(par_dir=tmp_path, start_server=False)

    def test_interleaved_pause_continue(
        self,
        tracker: TensorBoard,
        interleaved_workflow: tuple[log_events.Event, ...],
        run_a_dir: Callable[[str], pathlib.Path],
        run_b_dir: Callable[[str], pathlib.Path],
        example_model_name: str,
        example_source_name: str,
    ) -> None:
        """Verify TensorBoard writer stashing with interleaved runs."""
        # Trigger
        for event in interleaved_workflow:
            tracker.notify(event)
        tracker.close()

        # Assert
        dir_a = run_a_dir(TensorBoard.folder_name)
        dir_b = run_b_dir(TensorBoard.folder_name)

        files_a = list(dir_a.rglob('events.out.tfevents*'))
        files_b = list(dir_b.rglob('events.out.tfevents*'))

        acc_a = EventAccumulator(str(dir_a))
        acc_a.Reload()
        tag = f'{example_model_name}/{example_source_name}-loss'
        scalars_a = [round(e.value, 4) for e in acc_a.Scalars(tag)]

        acc_b = EventAccumulator(str(dir_b))
        acc_b.Reload()
        scalars_b = [round(e.value, 4) for e in acc_b.Scalars(tag)]

        assert dir_a.exists()
        assert dir_b.exists()
        assert len(files_a) >= 1
        assert len(files_b) >= 1
        assert all(f.stat().st_size > 0 for f in files_a)
        assert all(f.stat().st_size > 0 for f in files_b)
        assert scalars_a == [0.1, 0.05]
        assert scalars_b == [0.9]
        assert 0.9 not in scalars_a

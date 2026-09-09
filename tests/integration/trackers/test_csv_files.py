"""Tests for CSVDumper interleaved pause and continue."""

import csv
import pathlib

from collections.abc import Callable

import pytest

from drytorch.core import log_events
from drytorch.trackers.csv import CSVDumper


class TestCSVDumperPauseContinue:
    """Tests CSVDumper state isolation across interleaved runs."""

    @pytest.fixture
    def tracker(self, tmp_path: pathlib.Path) -> CSVDumper:
        """Create a CSVDumper instance."""
        return CSVDumper(tmp_path)

    def test_interleaved_pause_continue(
        self,
        tracker: CSVDumper,
        interleaved_workflow: tuple[log_events.Event, ...],
        run_a_dir: Callable[[str], pathlib.Path],
        run_b_dir: Callable[[str], pathlib.Path],
        example_model_name: str,
        example_source_name: str,
    ) -> None:
        """Verify that CSVDumper keeps run A and B metrics isolated."""
        # Trigger
        for event in interleaved_workflow:
            tracker.notify(event)
        tracker.close()

        # Assert
        dir_a = run_a_dir(CSVDumper.folder_name)
        dir_b = run_b_dir(CSVDumper.folder_name)

        csv_a = dir_a / example_model_name / f'{example_source_name}.csv'
        csv_b = dir_b / example_model_name / f'{example_source_name}.csv'

        with csv_a.open(newline='') as f:
            rows_a = list(csv.reader(f))
        with csv_b.open(newline='') as f:
            rows_b = list(csv.reader(f))

        loss_idx_a = rows_a[0].index('loss')
        loss_values_a = [float(row[loss_idx_a]) for row in rows_a[1:]]

        loss_idx_b = rows_b[0].index('loss')
        loss_values_b = [float(row[loss_idx_b]) for row in rows_b[1:]]

        assert csv_a.exists()
        assert csv_b.exists()
        assert loss_values_a == [0.1, 0.05]
        assert loss_values_b == [0.9]
        assert 0.9 not in loss_values_a

"""Tests for the files that recorded and unrecorded runs leave."""

import pathlib

import pytest

from drytorch.core.experimenting import Experiment


try:
    from drytorch.trackers.yaml import MetadataDumper
except ImportError:
    pytest.skip('yaml not available', allow_module_level=True)
    raise


class TestRunRecording:
    """Test that only recorded runs reach the persistent trackers."""

    @pytest.fixture
    def experiment(self, tmp_path: pathlib.Path) -> Experiment[dict[str, int]]:
        """Set up an experiment that dumps its metadata."""
        experiment = Experiment({'n_epochs': 1}, name='rec', par_dir=tmp_path)
        experiment.trackers.subscribe(test_metadata=MetadataDumper())
        return experiment

    def test_unrecorded_run_writes_no_metadata(
        self, experiment: Experiment[dict[str, int]], tmp_path: pathlib.Path
    ) -> None:
        """Test an unrecorded run leaves no metadata."""
        with experiment.create_run(record=False):
            pass

        assert not (tmp_path / MetadataDumper.folder_name).exists()

    def test_recorded_run_writes_metadata(
        self, experiment: Experiment[dict[str, int]], tmp_path: pathlib.Path
    ) -> None:
        """Test a recorded run dumps its configuration."""
        with experiment.create_run():
            pass

        exp_dir = tmp_path / MetadataDumper.folder_name / experiment.name
        assert list(exp_dir.rglob('config_*.yaml'))

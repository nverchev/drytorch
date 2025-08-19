"""Tests HydraLink integrates into hydra mode tracker's initialization."""

import pytest


try:
    import hydra

    from omegaconf import DictConfig
except ImportError:
    pytest.skip('hydra not available', allow_module_level=True)
    raise

import datetime
import logging
import pathlib
import sys

import drytorch

from drytorch.trackers.hydra import HydraLink
from drytorch.trackers.logging import BuiltinLogger
from drytorch.trackers.tqdm import TqdmLogger


expected_path_folder = pathlib.Path(__file__).parent / 'expected_logs'
expected_log = expected_path_folder / 'hydra_log_file.txt'
expected_out = expected_path_folder / 'hydra_out_file.txt'


class TestHydraFullCycle:
    """Complete HydraLink session and tests it afterward."""

    # TODO: full_cycle fixture's scope should be changed to "class"
    @pytest.fixture(autouse=True)
    def full_cycle(
        self,
        capsys,
        tmp_path_factory,
        monkeypatch,
        start_experiment_event,
        metrics_event,
        iterate_batch_event,
        stop_experiment_event,
    ) -> None:
        """Setup test environment with actual hydra configuration."""
        self.hydra_dir = tmp_path_factory.mktemp('outputs')
        run_dir_arg = f'++hydra.run.dir={self.hydra_dir.as_posix()}'

        def _mock_format_time(*_, **__):
            fixed_time = datetime.datetime(2024, 1, 1, 12)
            return fixed_time.strftime('%Y-%m-%d %H:%M:%S')

        with monkeypatch.context() as m:
            # fix timestamp for reproducibility
            m.setattr(logging.Formatter, 'formatTime', _mock_format_time)
            m.setattr(sys, 'argv', ['test_script', run_dir_arg])

            @hydra.main(version_base=None)
            def _app(_: DictConfig):
                drytorch.init_trackers(mode='hydra')
                trackers = (
                    BuiltinLogger(),
                    TqdmLogger(leave=False),
                    HydraLink(),
                )
                events = (
                    start_experiment_event,
                    metrics_event,
                    iterate_batch_event,
                    stop_experiment_event,
                )

                for event in events:
                    for tracker in trackers:
                        tracker.notify(event)
                return

            _app()
        return

    def test_console_logging(self, capsys) -> None:
        """Test HydraLink logs to stdout are deduplicated."""
        hydra_out = capsys.readouterr().out.strip().expandtabs(4)
        hydra_out_cleaned = hydra_out.replace('\n\r', '\n')

        with expected_out.open() as file:
            expected_file_out = file.read()

        assert hydra_out_cleaned == expected_file_out.strip()

    def test_log_file(self, start_experiment_event) -> None:
        """Test HydraLink creates file log with expected format."""
        file_name = pathlib.Path(__file__).name
        log_file = self.hydra_dir / file_name

        with expected_log.open() as file:
            expected_file_log = file.read()

        with log_file.with_suffix('.log').open() as hydra_file:
            hydra_log = hydra_file.read().strip().expandtabs(4)

        assert hydra_log == expected_file_log.strip()

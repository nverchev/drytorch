"""Testing local reorganization of hydra folders."""

import sys

from collections.abc import Generator

import pytest


try:
    import hydra

    from omegaconf import DictConfig
except ImportError:
    pytest.skip('hydra not available', allow_module_level=True)
    raise

from drytorch.trackers.hydra import HydraLink


class TestHydraLinkFullCycle:
    """Complete HydraLink session and tests it afterward."""

    @pytest.fixture()
    def tracker(
        self,
        tmp_path_factory,
        monkeypatch,
        start_experiment_event,
        stop_experiment_event,
    ) -> Generator[HydraLink, None, None]:
        """Setup test environment with actual hydra configuration."""
        self.hydra_dir = tmp_path_factory.mktemp('outputs')
        run_dir_arg = f'++hydra.run.dir={self.hydra_dir.as_posix()}'
        tracker: HydraLink | None = None

        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['test_script', run_dir_arg])

            @hydra.main(version_base=None)
            def _app(_: DictConfig) -> None:
                nonlocal tracker
                tracker = HydraLink()
                return

            _app()
            if tracker is None:
                raise RuntimeError('Setup failed')

            tracker.notify(start_experiment_event)
            yield tracker

            tracker.notify(stop_experiment_event)
            return

    def test_log_file(self, tracker) -> None:
        """Test HydraLink moved the files to the correct location."""
        assert list(tracker._get_run_dir().iterdir())

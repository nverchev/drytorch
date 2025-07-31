"""Testing local reorganization of hydra folders."""

import sys

import pytest


try:
    import hydra

    from omegaconf import DictConfig
except ImportError:
    pytest.skip('hydra not available', allow_module_level=True)
    raise


from drytorch import Experiment
from drytorch.trackers.hydra import HydraLink


class TestHydraLinkFullCycle:
    """Complete HydraLink session and tests it afterward."""

    # pylint: disable=attribute-defined-outside-init
    @pytest.fixture(autouse=True)
    def setup_full_cycle(self, tmp_path_factory, monkeypatch) -> None:
        """Setup test environment with actual hydra configuration."""
        self.hydra_dir = tmp_path_factory.mktemp('outputs')
        run_dir_arg = f'++hydra.run.dir={self.hydra_dir.as_posix()}'
        self.exp = Experiment(
            config=None, par_dir=tmp_path_factory.mktemp('exp')
        )
        self.exp.trackers.named_trackers.clear()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['test_script', run_dir_arg])

            @hydra.main(version_base=None)
            def _app(_: DictConfig) -> None:
                self.exp.trackers.register(HydraLink())
                with self.exp:
                    pass

                return

            _app()
        return

    def test_log_file(self) -> None:
        """Test HydraLink creates file log with expected format."""
        hydra_runs = self.exp.dir / HydraLink.hydra_folder
        assert list(hydra_runs.iterdir())

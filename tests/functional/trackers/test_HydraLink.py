"""Functional tests for HydraLink."""

import pytest
try:
    import hydra
    from omegaconf import DictConfig
except ImportError:
    pytest.skip('hydra not available', allow_module_level=True)
    raise

import sys

from drytorch import Experiment
from drytorch.trackers.hydra import HydraLink


class TestHydraLinkFullCycle:
    """Complete HydraLink session and tests it afterward."""

    @pytest.fixture(autouse=True)
    def full_cycle(self,
                   tmp_path_factory,
                   monkeypatch) -> None:
        """Setup test environment with actual hydra configuration."""
        self.hydra_dir = tmp_path_factory.mktemp('outputs')
        run_dir_arg = f'++hydra.run.dir={self.hydra_dir.as_posix()}'
        self.exp = Experiment[None](par_dir=tmp_path_factory.mktemp('exp'))
        self.exp.trackers.named_trackers.clear()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['test_script', run_dir_arg])

            @hydra.main(version_base=None)
            def _app(_: DictConfig):
                self.exp.trackers.register(HydraLink())
                with self.exp:
                    pass

                return

            _app()
        return

    def test_log_file(self) -> None:
        """Test HydraLink creates file log with expected format."""
        exp_dir = self.exp.dir
        hydra_folder = HydraLink.hydra_folder
        link_name = HydraLink.link_name
        linked_folder = exp_dir / hydra_folder / link_name
        assert linked_folder.exists()

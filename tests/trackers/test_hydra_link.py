"""Test Hydra Link tracker"""

import pytest

import pathlib
import shutil
from typing import Generator

import hydra
from omegaconf import DictConfig

from dry_torch.trackers.hydra import HydraLink
from dry_torch import log_events

@pytest.mark.skip
class TestHydraLink:
    """Tests for the HydraLink tracker with actual Hydra integration."""

    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        """Setup test environment with actual hydra configuration."""
        self.base_dir = pathlib.Path(__file__).parent

        # Initialize config directory
        config_dir = self.base_dir / "conf"
        config_dir.mkdir(parents=True)

        # Create basic config file
        config_yaml = """
        defaults:
          - _self_
        output_dir: ${hydra:runtime.output_dir}
        """
        with open(config_dir / 'config.yaml', 'w') as f:
            f.write(config_yaml)
        with open(config_dir / '__init__.py', 'w') as f:
            f.write('')
        yield

        try:
            shutil.rmtree(config_dir)
        except FileNotFoundError:
            pass

    def test_initialization(self) -> None:
        """Test HydraLink initialization with actual Hydra config."""

        @hydra.main(config_path="conf", config_name="config", version_base=None)
        def _app(_cfg: DictConfig) -> None:
            tracker = HydraLink(par_dir=self.base_dir)
            assert isinstance(tracker.par_dir, pathlib.Path)
            assert isinstance(tracker.hydra_dir, pathlib.Path)
            assert tracker._exp_dir is None
            assert tracker._counter == 0

        _app()

    def test_dir_access_before_experiment(self) -> None:
        """Test accessing dir property before experiment starts raises error."""

        @hydra.main(config_path="conf", config_name="config", version_base=None)
        def test_app(_cfg: DictConfig) -> None:
            tracker = HydraLink(par_dir=self.base_dir)
            with pytest.raises(RuntimeError,
                               match="Accessed outside experiment scope"):
                _ = tracker.dir

        test_app()

    def test_start_experiment(
            self,
            start_experiment_event: log_events.StartExperiment
    ) -> None:
        """Test handling of StartExperiment event with actual Hydra.

        Args:
            start_experiment_event: StartExperiment event fixture
        """

        @hydra.main(config_path="conf", config_name="config", version_base=None)
        def test_app(_cfg: DictConfig) -> None:
            tracker = HydraLink(par_dir=self.base_dir)
            tracker.notify(start_experiment_event)

            # Check if experiment directory is set
            assert tracker._exp_dir == start_experiment_event.exp_dir

            # Verify symlink creation
            symlink_path = self.base_dir / ".hydra"
            assert symlink_path.is_symlink()
            assert symlink_path.resolve() == tracker.hydra_dir

        test_app()

    def test_multiple_start_experiments(
            self,
            start_experiment_event: log_events.StartExperiment
    ) -> None:
        """Test handling multiple StartExperiment events with actual Hydra.

        Args:
            start_experiment_event: StartExperiment event fixture
        """

        @hydra.main(config_path="conf", config_name="config", version_base=None)
        def test_app(_cfg: DictConfig) -> None:
            tracker = HydraLink(par_dir=self.base_dir)

            # First experiment
            tracker.notify(start_experiment_event)
            first_link = self.base_dir / ".hydra"
            assert first_link.is_symlink()

            # Second experiment (should create .hydra_1)
            tracker.notify(start_experiment_event)
            second_link = self.base_dir / ".hydra_1"
            assert second_link.is_symlink()

            assert tracker._counter == 1

        test_app()

    def test_stop_experiment(
            self,
            start_experiment_event: log_events.StartExperiment,
            stop_experiment_event: log_events.StopExperiment
    ) -> None:
        """Test handling of StopExperiment event with actual Hydra.

        Args:
            start_experiment_event: StartExperiment event fixture
            stop_experiment_event: StopExperiment event fixture
        """

        @hydra.main(config_path="conf", config_name="config", version_base=None)
        def test_app(_cfg: DictConfig) -> None:
            tracker = HydraLink(par_dir=self.base_dir)

            # Start experiment first
            tracker.notify(start_experiment_event)
            initial_link = self.base_dir / ".hydra"

            # Create some test files in Hydra directory
            (tracker.hydra_dir / "config.yaml").touch()
            (tracker.hydra_dir / "output.log").touch()

            # Stop experiment
            tracker.notify(stop_experiment_event)

            # Verify symlink is replaced with directory
            assert not initial_link.is_symlink()
            assert initial_link.is_dir()

            # Verify content is copied
            assert (initial_link / "config.yaml").exists()
            assert (initial_link / "output.log").exists()

            # Verify experiment directory is reset
            assert tracker._exp_dir is None

        test_app()

    def test_error_when_hydra_not_started(self) -> None:
        """Test initialization fails when Hydra directory doesn't exist."""
        # Temporarily clear Hydra's initialized state
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type: ignore

        with pytest.raises(RuntimeError, match="Hydra has not started"):
            # Try to initialize tracker without Hydra context
            HydraLink(par_dir=self.base_dir)

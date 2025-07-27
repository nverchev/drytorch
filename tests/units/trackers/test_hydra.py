"""Tests for the "hydra" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('hydra'):
    pytest.skip('hydra not available', allow_module_level=True)
from drytorch import exceptions
from drytorch.trackers.hydra import HydraLink


class TestHydraLink:
    """Tests for the HydraLink tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker, tmp_path) -> None:
        """Setup test environment."""
        self.hydra_output_dir = (tmp_path / 'outputs').resolve()
        self.hydra_output_dir.mkdir()
        mock_config = mocker.MagicMock()
        mock_config.runtime.output_dir = self.hydra_output_dir.as_posix()
        mocker.patch(
            'hydra.core.hydra_config.HydraConfig.get', return_value=mock_config
        )
        return

    @pytest.fixture
    def tracker(self, tmp_path) -> HydraLink:
        """Set up the instance."""
        return HydraLink(par_dir=tmp_path)

    @pytest.fixture
    def tracker_no_copy(self, tmp_path) -> HydraLink:
        """Set up the instance with copy_hydra=False."""
        return HydraLink(par_dir=tmp_path, copy_hydra=False)

    def test_cleanup_no_copy(
        self, tracker_no_copy, start_experiment_mock_event
    ):
        """Test cleanup does not copy hydra folder."""
        tracker_no_copy.notify(start_experiment_mock_event)
        tracker_no_copy.clean_up()
        assert tracker_no_copy.dir.is_symlink()

    def test_cleanup_with_copy(self, tracker, start_experiment_mock_event):
        """Test cleanup copies hydra folder."""
        tracker.notify(start_experiment_mock_event)
        tracker.clean_up()
        assert tracker.dir.is_dir()

    def test_init_with_valid_hydra(self, tracker, tmp_path) -> None:
        """Test initialization with a valid Hydra configuration."""
        assert tracker.par_dir == tmp_path
        assert isinstance(tracker.hydra_folder, str)
        assert tracker.hydra_dir == self.hydra_output_dir

    def test_init_without_hydra_raises_exception(self, tmp_path) -> None:
        """Test initialization fails with a non-existing hydra directory."""
        self.hydra_output_dir.rmdir()
        with pytest.raises(exceptions.TrackerError):
            HydraLink(par_dir=tmp_path / 'not_existing')

    def test_dir_property(self, tracker, tmp_path) -> None:
        """Test dir property returns the same directory twice."""
        dir1 = tracker.dir
        dir2 = tracker.dir
        assert dir1 == dir2

    def test_notify_start_experiment_creates_symlink(
        self,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test start experiment notification creates symlink."""
        tracker.notify(start_experiment_mock_event)
        assert tracker.dir.is_symlink()
        assert tracker.dir.resolve() == self.hydra_output_dir

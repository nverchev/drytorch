"""Tests for the "hydra" module."""

import pytest

from dry_torch import exceptions
from dry_torch.trackers.hydra import HydraLink


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
            'hydra.core.hydra_config.HydraConfig.get',
            return_value=mock_config
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

    def test_cleanup(self, tracker_no_copy):
        tracker_no_copy.clean_up()
        assert tracker_no_copy._counter == 0

    def test_init_with_valid_hydra(self, tracker, tmp_path) -> None:
        """Test initialization with valid Hydra configuration."""
        assert tracker.par_dir == tmp_path
        assert isinstance(tracker.hydra_folder, str)
        assert isinstance(tracker.link_name, str)
        assert tracker.hydra_dir == self.hydra_output_dir

    def test_init_without_hydra_raises_exception(self, tmp_path) -> None:
        """Test initialization fails without existing Hydra output directory."""
        self.hydra_output_dir.rmdir()
        with pytest.raises(exceptions.TrackerException):
            HydraLink(par_dir=tmp_path / 'not_existing')

    def test_dir_property(self, tracker, tmp_path) -> None:
        """Test dir property returns correct path with counter."""
        tracker._counter = 2
        hydra_path = tmp_path / tracker.hydra_folder
        expected_dir = hydra_path / f'{tracker.link_name}_2'
        assert tracker.dir == expected_dir
        assert hydra_path.exists()

    def test_notify_start_experiment_creates_symlink(
            self,
            tracker,
            start_experiment_mock_event,
    ) -> None:
        """Test start experiment notification creates symlink."""
        tracker.notify(start_experiment_mock_event)
        assert tracker.dir.is_symlink()
        assert tracker.dir.resolve() == self.hydra_output_dir

    def test_notify_start_experiment_handles_existing_link(
            self,
            tracker,
            start_experiment_mock_event,
    ) -> None:
        """Test counter increments when existing symlink."""
        # Create existing link
        link_dir = tracker.dir
        link_dir.symlink_to(tracker.hydra_dir, target_is_directory=True)
        tracker.notify(start_experiment_mock_event)
        assert tracker._counter == 1

    def test_notify_stop_experiment_with_copy(
            self,
            tracker,
            start_experiment_mock_event,
            stop_experiment_mock_event,
    ) -> None:
        """Test stop experiment notification copies hydra folder."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(stop_experiment_mock_event)
        assert tracker.dir.is_dir()
        assert tracker._exp_dir is None
        assert tracker._counter == 0

    def test_notify_stop_experiment_without_copy(
            self,
            tracker_no_copy,
            start_experiment_mock_event,
            stop_experiment_mock_event,
    ) -> None:
        """Test stop experiment notification without copying."""
        tracker_no_copy.notify(start_experiment_mock_event)
        tracker_no_copy.notify(stop_experiment_mock_event)
        assert tracker_no_copy.dir.is_symlink()
        assert tracker_no_copy._exp_dir is None
        assert tracker_no_copy._counter == 0

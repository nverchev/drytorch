"""Tests for the "hydra" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('hydra'):
    pytest.skip('hydra not available', allow_module_level=True)

from drytorch.core import exceptions
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
        """Test cleanup does not copy the hydra folder."""
        tracker_no_copy.notify(start_experiment_mock_event)
        link = tracker_no_copy._get_run_dir()
        tracker_no_copy.clean_up()
        assert link.is_symlink()

    def test_cleanup_with_copy(self, tracker, start_experiment_mock_event):
        """Test cleanup copies hydra folder."""
        tracker.notify(start_experiment_mock_event)
        link = tracker._get_run_dir()
        tracker.clean_up()
        assert link.is_dir()

    def test_init_with_valid_hydra(self, tracker, tmp_path) -> None:
        """Test initialization with a valid Hydra configuration."""
        assert tracker.par_dir == tmp_path
        assert isinstance(tracker.folder_name, str)
        assert tracker.hydra_dir == self.hydra_output_dir

    def test_init_without_hydra_raises_exception(self, tmp_path) -> None:
        """Test initialization fails with a non-existing hydra directory."""
        self.hydra_output_dir.rmdir()
        with pytest.raises(exceptions.TrackerError):
            HydraLink(par_dir=tmp_path / 'not_existing')

    def test_notify_start_experiment_creates_symlink(
        self,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test start experiment notification creates symlink."""
        tracker.notify(start_experiment_mock_event)
        assert tracker._get_run_dir().is_symlink()
        assert tracker._get_run_dir().resolve() == self.hydra_output_dir

    def test_init_when_hydra_config_not_set_raises_tracker_error(
        self, mocker, tmp_path
    ) -> None:
        """Test init raises TrackerError when HydraConfig was not set."""
        mocker.patch(
            'hydra.core.hydra_config.HydraConfig.get',
            side_effect=ValueError('HydraConfig was not set'),
        )
        with pytest.raises(
            exceptions.TrackerError, match='Hydra has not started'
        ):
            HydraLink(par_dir=tmp_path)

    def test_notify_start_experiment_same_timestamp_collision(
        self, tmp_path, start_experiment_mock_event
    ) -> None:
        """Test multiple runs starting in the same second avoid collisions."""
        t1 = HydraLink(par_dir=tmp_path)
        t1.notify(start_experiment_mock_event)
        link1 = t1._get_run_dir()
        assert link1.is_symlink()

        t2 = HydraLink(par_dir=tmp_path)
        t2.notify(start_experiment_mock_event)
        link2 = t2._get_run_dir()
        assert link2.is_symlink()
        assert link2 != link1
        assert link2.name == f'{link1.name}_1'
        assert link2.resolve() == self.hydra_output_dir

    def test_notify_start_experiment_collision_after_cleanup(
        self, tmp_path, start_experiment_mock_event
    ) -> None:
        """Test collision resolution after run was converted to directory."""
        t1 = HydraLink(par_dir=tmp_path, copy_hydra=True)
        t1.notify(start_experiment_mock_event)
        dir1 = t1._get_run_dir()
        t1.clean_up()
        assert dir1.is_dir()
        assert not dir1.is_symlink()

        t2 = HydraLink(par_dir=tmp_path)
        t2.notify(start_experiment_mock_event)
        link2 = t2._get_run_dir()
        assert link2.is_symlink()
        assert link2.name == f'{dir1.name}_1'

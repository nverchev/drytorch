"""Tests for the "yaml" module."""

import importlib.util

from collections.abc import Generator

import pytest


if not importlib.util.find_spec('yaml'):
    pytest.skip('yaml not available', allow_module_level=True)

from drytorch.trackers.yaml import YamlDumper


class TestYamlDumper:
    """Tests for the YamlDumper tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
        self.mock_dump = mocker.patch('yaml.dump')
        return

    @pytest.fixture
    def tracker(self, tmp_path) -> YamlDumper:
        """Set up the instance."""
        return YamlDumper(par_dir=tmp_path)

    @pytest.fixture
    def tracker_started(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[YamlDumper, None, None]:
        """Set up the instance with resume."""
        tracker.notify(start_experiment_mock_event)
        yield tracker
        tracker.notify(stop_experiment_mock_event)
        return

    def test_class_attributes(self) -> None:
        """Test class attributes' existence."""
        assert isinstance(YamlDumper.folder_name, str)

    def test_notify_configuration(self, tracker_started) -> None:
        """Test notification of a model registration event."""
        self.mock_dump.assert_called_once()

    def test_notify_model_registration(
        self, tracker_started, model_registration_mock_event
    ) -> None:
        """Test notification of a model registration event."""
        self.mock_dump.assert_called_once()
        self.mock_dump.reset_mock()
        tracker_started.notify(model_registration_mock_event)
        self.mock_dump.assert_called_once()

    def test_notify_actor_registration(
        self, tracker_started, actor_registration_mock_event
    ) -> None:
        """Test notification of a source registration event."""
        self.mock_dump.assert_called_once()
        self.mock_dump.reset_mock()
        tracker_started.notify(actor_registration_mock_event)
        # metadata dumped in the metadata folder and in the archive folder
        self.mock_dump.assert_called_once()

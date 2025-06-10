"""Tests for the "yaml" module."""

import pytest

from typing import Generator

from dry_torch.trackers.yaml import YamlDumper


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
        assert isinstance(YamlDumper.metadata_folder, str)
        assert isinstance(YamlDumper.archive_folder, str)

    def test_notify_model_creation(self,
                                   tracker_started,
                                   model_creation_mock_event) -> None:
        """Test notification of model creation event."""
        tracker_started.notify(model_creation_mock_event)
        # metadata dumped in metadata folder and in the archive folder
        assert self.mock_dump.call_count == 2

    def test_notify_call_model(self,
                               tracker_started,
                               call_model_mock_event) -> None:
        """Test notification of call model event."""
        tracker_started.notify(call_model_mock_event)
        # metadata dumped in metadata folder and in the archive folder
        assert self.mock_dump.call_count == 2

    def test_version_method_creates_directories(self,
                                                tracker,
                                                tmp_path) -> None:
        """Test that _version method creates correct directory structure."""
        metadata = {'key': 'value'}
        sub_folder = 'test_model'
        file_name = 'model_file'
        file_version = 'v1.0'
        tracker._version(metadata, sub_folder, file_name, file_version)
        metadata_path = tmp_path / sub_folder / YamlDumper.metadata_folder
        archived_path = metadata_path / YamlDumper.archive_folder
        assert metadata_path.exists()
        assert archived_path.exists()

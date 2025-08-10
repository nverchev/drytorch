"""Functional tests for YamlDumper."""

import pytest


try:
    import yaml
except ImportError:
    pytest.skip('yaml not available', allow_module_level=True)
    raise

from drytorch.trackers.yaml import YamlDumper


class TestSQLConnectionFullCycle:
    """Complete SQLConnection session and tests it afterward."""

    @pytest.fixture
    def tracker(self, tmp_path) -> YamlDumper:
        """Set up the instance."""
        return YamlDumper(tmp_path)

    def test_config_metadata(self,
                             tracker,
                             start_experiment_event,
                             example_config):
        """Test correct dumping off config metadata."""
        tracker.notify(start_experiment_event)
        address = tracker._get_run_dir() / 'config.yaml'
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_config

    def test_model_metadata(self,
                            tracker,
                            start_experiment_event,
                            model_registration_event,
                            example_metadata):
        """Test correct dumping of metadata from the model."""
        tracker.notify(start_experiment_event)
        model_name = model_registration_event.model_name
        metadata_path = tracker._get_run_dir() / model_name
        address = metadata_path / model_name
        tracker.notify(model_registration_event)
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_metadata

    def test_caller_metadata(self,
                             tracker,
                             start_experiment_event,
                             source_registration_event,
                             example_metadata):
        """Test correct dumping of metadata from the caller."""
        tracker.notify(start_experiment_event)
        model_name = source_registration_event.model_name
        source_name = source_registration_event.source_name
        metadata_path = tracker._get_run_dir() / model_name
        address = metadata_path / source_name
        tracker.notify(source_registration_event)
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_metadata

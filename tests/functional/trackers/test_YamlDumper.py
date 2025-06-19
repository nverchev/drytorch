"""Functional tests for YamlDumper."""

import pytest
try:
    import yaml  # type ignore
except ImportError:
    pytest.skip('yaml not available', allow_module_level=True)
    raise

from drytorch.trackers.yaml import YamlDumper


class TestSQLConnectionFullCycle:
    """Complete SQLConnection session and tests it afterward."""

    @pytest.fixture
    def tracker(self) -> YamlDumper:
        """Set up the instance."""
        return YamlDumper()

    @pytest.fixture(autouse=True)
    def full_cycle(self, tracker, event_workflow) -> None:
        """Run an example session."""
        for event in event_workflow:
            tracker.notify(event)
        return

    def test_model_metadata(self,
                            start_experiment_event,
                            model_creation_event,
                            example_metadata):
        par_dir = start_experiment_event.exp_dir
        model_name = model_creation_event.model_name
        model_version = model_creation_event.model_version
        metadata_path = par_dir / model_name / YamlDumper.metadata_folder
        address = metadata_path / model_name
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_metadata

        archive_folder = metadata_path / YamlDumper.archive_folder / model_name
        archive_address = archive_folder / model_version
        with archive_address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_metadata

    def test_caller_metadata(self,
                             start_experiment_event,
                             call_model_event,
                             example_metadata):
        par_dir = start_experiment_event.exp_dir
        model_name = call_model_event.model_name
        file_name = call_model_event.source_name
        source_version = call_model_event.source_version
        metadata_path = par_dir / model_name / YamlDumper.metadata_folder
        address = metadata_path / file_name
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_metadata

        archive_folder = metadata_path / YamlDumper.archive_folder / file_name
        archive_address = archive_folder / source_version
        with archive_address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_metadata

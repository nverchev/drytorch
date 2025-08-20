"""Functional tests for YamlDumper."""

import pytest


try:
    import yaml
except ImportError:
    pytest.skip('yaml not available', allow_module_level=True)
    raise

from drytorch.trackers.yaml import YamlDumper
from drytorch.utils import repr_utils


TS_FMT = repr_utils.CreatedAtMixin.ts_fmt


class TestSQLConnectionFullCycle:
    """Complete SQLConnection session and tests it afterward."""

    @pytest.fixture
    def tracker(self, tmp_path) -> YamlDumper:
        """Set up the instance."""
        return YamlDumper(tmp_path)

    def test_config_metadata(
        self, tracker, start_experiment_event, example_run_ts, example_config
    ):
        """Test correct dumping off config metadata."""
        tracker.notify(start_experiment_event)
        example_run_ts_str = example_run_ts.strftime(TS_FMT)
        address = tracker._get_run_dir() / f'config_{example_run_ts_str}.yaml'
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_config

    def test_model_metadata(
        self,
        tracker,
        start_experiment_event,
        model_registration_event,
        example_model_name,
        example_model_ts,
        example_architecure_repr,
    ):
        """Test correct dumping of metadata from the model."""
        tracker.notify(start_experiment_event)
        example_model_ts_str = example_model_ts.strftime(TS_FMT)
        metadata_folder = f'{example_model_name}_{example_model_ts_str}'
        metadata_path = tracker._get_run_dir() / metadata_folder
        address = metadata_path / 'architecture'
        tracker.notify(model_registration_event)
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_architecure_repr

    def test_caller_metadata(
        self,
        tracker,
        start_experiment_event,
        actor_registration_event,
        example_model_name,
        example_model_ts,
        example_source_name,
        example_source_ts,
        example_metadata,
    ):
        """Test correct dumping of metadata from the caller."""
        tracker.notify(start_experiment_event)
        example_model_ts_str = example_model_ts.strftime(TS_FMT)
        metadata_folder = f'{example_model_name}_{example_model_ts_str}'
        metadata_path = tracker._get_run_dir() / metadata_folder
        example_source_ts_str = example_source_ts.strftime(TS_FMT)
        address = metadata_path / (
            f'{example_source_name}_{example_source_ts_str}'
        )
        tracker.notify(actor_registration_event)
        with address.with_suffix('.yaml').open() as file:
            metadata = yaml.safe_load(file)

        assert metadata == example_metadata

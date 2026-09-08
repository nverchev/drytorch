"""Tests for the "yaml" module."""

import importlib.util

from collections.abc import Generator

import pytest


if not importlib.util.find_spec('yaml'):
    pytest.skip('yaml not available', allow_module_level=True)

import yaml

from drytorch.trackers import yaml as yaml_tracker
from drytorch.trackers.yaml import (
    DryTorchDumper,
    MetadataDumper,
    has_short_repr,
)


class TestMetadataDumper:
    """Tests for the MetadataDumper tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
        self.mock_dump = mocker.patch('yaml.dump')
        return

    @pytest.fixture
    def tracker(self, tmp_path) -> MetadataDumper:
        """Set up the instance."""
        return MetadataDumper(par_dir=tmp_path)

    @pytest.fixture
    def tracker_started(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[MetadataDumper, None, None]:
        """Set up the instance with resume."""
        tracker.notify(start_experiment_mock_event)
        yield tracker
        tracker.notify(stop_experiment_mock_event)
        return

    def test_class_attributes(self) -> None:
        """Test class attributes' existence."""
        assert isinstance(MetadataDumper.folder_name, str)

    def test_notify_configuration(self, tracker_started) -> None:
        """Test notification of a model registration event."""
        self.mock_dump.assert_called_once()
        _, kwargs = self.mock_dump.call_args
        assert kwargs.get('Dumper') is DryTorchDumper

    def test_notify_model_registration(
        self, tracker_started, model_registration_mock_event
    ) -> None:
        """Test notification of a model registration event."""
        self.mock_dump.assert_called_once()
        self.mock_dump.reset_mock()
        tracker_started.notify(model_registration_mock_event)
        self.mock_dump.assert_called_once()
        _, kwargs = self.mock_dump.call_args
        assert kwargs.get('Dumper') is DryTorchDumper

    def test_notify_actor_registration(
        self, tracker_started, actor_registration_mock_event
    ) -> None:
        """Test notification of a source registration event."""
        self.mock_dump.assert_called_once()
        self.mock_dump.reset_mock()
        tracker_started.notify(actor_registration_mock_event)
        # metadata dumped in the metadata folder and in the archive folder
        self.mock_dump.assert_called_once()
        _, kwargs = self.mock_dump.call_args
        assert kwargs.get('Dumper') is DryTorchDumper


class TestYamlSerialization:
    """Tests for YAML serialization settings and scoping."""

    def test_global_pyyaml_unaffected(self) -> None:
        """Test that importing drytorch does not mutate global PyYAML."""
        dumped_tuple = yaml.dump({'t': (1, 2)})
        assert '!!python/tuple' in dumped_tuple
        dumped_set = yaml.dump({'s': {1, 2}})
        assert '!!set' in dumped_set

    def test_dynamic_length_constants(self, monkeypatch) -> None:
        """Test changing length constants affects serialization dynamically."""
        monkeypatch.setattr(yaml_tracker, 'MAX_LENGTH_PLAIN_REPR', 1)
        dumped = yaml.dump(['a', 'b'], Dumper=DryTorchDumper)
        assert '- a\n- b' in dumped

        monkeypatch.setattr(yaml_tracker, 'MAX_LENGTH_SHORT_REPR', 2)
        assert has_short_repr('abc') is False
        assert has_short_repr('ab') is True

    def test_set_dumping_is_deterministic(self) -> None:
        """Test that sets and frozensets are serialized in sorted order."""
        dumped = yaml.dump(
            {'tags': {'delta', 'gamma', 'alpha', 'beta'}},
            Dumper=DryTorchDumper,
        )
        assert '[alpha, beta, delta, gamma]' in dumped

        dumped_frozen = yaml.dump(
            {'tags': frozenset(['delta', 'gamma', 'alpha', 'beta'])},
            Dumper=DryTorchDumper,
        )
        assert '[alpha, beta, delta, gamma]' in dumped_frozen

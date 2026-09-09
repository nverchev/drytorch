"""Tests for YAML dumping, representer settings, and run isolation."""

import pathlib

from collections.abc import Callable

import pytest


try:
    from hypothesis import assume, given
    from hypothesis.strategies import characters, text
except ImportError:
    pytest.skip('hypothesis not available', allow_module_level=True)
    raise

try:
    import yaml
except ImportError:
    pytest.skip('yaml not available', allow_module_level=True)
    raise

from drytorch.core import log_events
from drytorch.trackers.yaml import (
    MAX_LENGTH_PLAIN_REPR,
    MAX_LENGTH_SHORT_REPR,
    DryTorchDumper,
    MetadataDumper,
    has_short_repr,
)
from drytorch.utils import repr_utils


def test_short_repr():
    """Test short_repr function."""
    assert has_short_repr('a' * MAX_LENGTH_SHORT_REPR) is True
    assert has_short_repr('a' * (MAX_LENGTH_SHORT_REPR + 1)) is False
    lit_str = repr_utils.LiteralStr('test')
    assert has_short_repr(lit_str) is False
    assert has_short_repr([]) is False  # false for other Sized objects
    assert has_short_repr(34) is True  # true for not a Sized object


def test_literal_string():
    """Test YAML representers for correct serialization."""
    str_value = 'test'
    lit_str = repr_utils.LiteralStr(str_value)
    yaml_output = yaml.dump(lit_str, Dumper=DryTorchDumper)
    assert yaml_output == yaml.dump(str_value, default_style='|')


def test_short_sequence():
    """Test sequence representation logic."""
    short_seq = ('a', 'b')
    yaml_output = yaml.dump(short_seq, Dumper=DryTorchDumper)
    assert yaml_output.strip() == '[a, b]'  # Flow style


def test_long_sequence():
    """Test sequence representation logic."""
    long_seq = ['a'] * (MAX_LENGTH_PLAIN_REPR + 1)
    yaml_output = yaml.dump(long_seq, Dumper=DryTorchDumper)
    assert '- ' in yaml_output  # Block style for long sequences


def test_long_element():
    """Test sequence representation logic."""
    long_element = ('a' * (MAX_LENGTH_SHORT_REPR + 1),)
    yaml_output = yaml.dump(long_element, Dumper=DryTorchDumper)
    assert '- ' in yaml_output  # Block style for long elements


def test_represent_omitted():
    """Test correct representation of omitted values."""
    omitted = repr_utils.Omitted(5)
    yaml_string = yaml.dump(omitted, Dumper=DryTorchDumper)
    assert yaml_string == '!Omitted\nomitted_elements: 5\n'


def test_represent_unknown_omitted():
    """Test correct representation of an unknown number of omitted values."""
    omitted = repr_utils.Omitted()
    yaml_string = yaml.dump(omitted, Dumper=DryTorchDumper)
    assert yaml_string == '!Omitted\nomitted_elements: .nan\n'


def test_represent_list_with_omitted():
    """Test the correct representation of omitted values inside a list."""
    yaml_string = yaml.dump(
        [2, repr_utils.Omitted(5), 3], Dumper=DryTorchDumper
    )
    assert yaml_string == '[2, !Omitted {omitted_elements: 5}, 3]\n'


@given(text(characters(codec='ascii', exclude_categories=['Cc', 'Cs'])))
def test_literal_str_yaml_representation(string):
    """Test LiteralStr is represented with the pipe style."""
    # pipe style incompatible with trailing spaces or empty strings
    stripped = string.strip()
    assume(stripped)
    literal = repr_utils.LiteralStr(stripped)
    yaml_literal = yaml.dump(literal, Dumper=DryTorchDumper)
    assert yaml_literal.startswith('|-\n')


class TestMetadataDumperPauseContinue:
    """Tests MetadataDumper metadata routing across interleaved runs."""

    @pytest.fixture
    def tracker(self, tmp_path: pathlib.Path) -> MetadataDumper:
        """Create a MetadataDumper instance."""
        return MetadataDumper(par_dir=tmp_path)

    def test_interleaved_pause_continue(
        self,
        tracker: MetadataDumper,
        start_experiment_event: log_events.StartExperimentEvent,
        pause_experiment_event: log_events.PauseExperimentEvent,
        continue_experiment_event: log_events.ContinueExperimentEvent,
        stop_experiment_event: log_events.StopExperimentEvent,
        start_experiment_event_b: log_events.StartExperimentEvent,
        stop_experiment_event_b: log_events.StopExperimentEvent,
        model_registration_event: log_events.ModelRegistrationEvent,
        actor_registration_event: log_events.ActorRegistrationEvent,
        run_a_dir: Callable[[str], pathlib.Path],
        run_b_dir: Callable[[str], pathlib.Path],
    ) -> None:
        """Verify MetadataDumper routes metadata to correct run directory."""
        # Trigger
        tracker.notify(start_experiment_event)
        tracker.notify(model_registration_event)
        tracker.notify(pause_experiment_event)

        tracker.notify(start_experiment_event_b)
        tracker.notify(model_registration_event)
        tracker.notify(stop_experiment_event_b)

        tracker.notify(continue_experiment_event)
        tracker.notify(actor_registration_event)
        tracker.notify(stop_experiment_event)
        tracker.close()

        # Assert
        dir_a = run_a_dir(MetadataDumper.folder_name)
        dir_b = run_b_dir(MetadataDumper.folder_name)

        files_a = list(dir_a.rglob('*.yaml'))
        files_b = list(dir_b.rglob('*.yaml'))

        config_a_file = next(dir_a.glob('config_*.yaml'))
        config_b_file = next(dir_b.glob('config_*.yaml'))

        with config_a_file.open() as f:
            config_a_content = yaml.safe_load(f)
        with config_b_file.open() as f:
            config_b_content = yaml.safe_load(f)

        actor_name = actor_registration_event.actor_name
        actor_files_a = list(dir_a.rglob(f'*{actor_name}*.yaml'))
        actor_files_b = list(dir_b.rglob(f'*{actor_name}*.yaml'))

        assert dir_a.exists()
        assert dir_b.exists()
        assert len(files_a) == 3
        assert len(files_b) == 2
        assert config_a_content == start_experiment_event.config
        assert config_b_content == start_experiment_event_b.config
        assert len(actor_files_a) == 1
        assert len(actor_files_b) == 0

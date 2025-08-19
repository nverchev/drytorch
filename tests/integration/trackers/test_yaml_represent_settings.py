"""Tests custom settings for YAML dumping."""

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

from drytorch.trackers.yaml import (
    MAX_LENGTH_PLAIN_REPR,
    MAX_LENGTH_SHORT_REPR,
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
    yaml_output = yaml.dump(lit_str)
    assert yaml_output == yaml.dump(str_value, default_style='|')


def test_short_sequence():
    """Test sequence representation logic."""
    short_seq = ('a', 'b')
    yaml_output = yaml.dump(short_seq)
    assert yaml_output.strip() == '[a, b]'  # Flow style


def test_long_sequence():
    """Test sequence representation logic."""
    long_seq = ['a'] * (MAX_LENGTH_PLAIN_REPR + 1)
    yaml_output = yaml.dump(long_seq)
    assert '- ' in yaml_output  # Block style for long sequences


def test_long_element():
    """Test sequence representation logic."""
    long_element = ('a' * (MAX_LENGTH_SHORT_REPR + 1),)
    yaml_output = yaml.dump(long_element)
    assert '- ' in yaml_output  # Block style for long elements


def test_represent_omitted():
    """Test correct representation of omitted values."""
    omitted = repr_utils.Omitted(5)
    yaml_string = yaml.dump(omitted)
    assert yaml_string == '!Omitted\nomitted_elements: 5\n'


def test_represent_unknown_omitted():
    """Test correct representation of an unknown number of omitted values."""
    omitted = repr_utils.Omitted()
    yaml_string = yaml.dump(omitted, Dumper=yaml.Dumper)
    assert yaml_string == '!Omitted\nomitted_elements: .nan\n'


def test_represent_list_with_omitted():
    """Test the correct representation of omitted values inside a list."""
    yaml_string = yaml.dump([2, repr_utils.Omitted(5), 3])
    assert yaml_string == '[2, !Omitted {omitted_elements: 5}, 3]\n'


@given(text(characters(codec='ascii', exclude_categories=['Cc', 'Cs'])))
def test_literal_str_yaml_representation(string):
    """Test LiteralStr is represented with the pipe style."""
    # pipe style incompatible with trailing spaces or empty strings
    stripped = string.strip()
    assume(stripped)
    literal = repr_utils.LiteralStr(stripped)
    yaml_literal = yaml.dump(literal)
    assert yaml_literal.startswith('|-\n')

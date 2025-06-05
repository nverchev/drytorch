"""Tests for the "yaml" module."""

from hypothesis import given, assume
from hypothesis.strategies import text, characters


import yaml  # type: ignore

from dry_torch.utils import repr_utils
from dry_torch.trackers.yaml import MAX_LENGTH_SHORT_REPR
from dry_torch.trackers.yaml import MAX_LENGTH_PLAIN_REPR
from dry_torch.trackers.yaml import has_short_repr


def test_short_repr():
    """Test short_repr function."""
    assert has_short_repr('a' * MAX_LENGTH_SHORT_REPR) is True
    assert has_short_repr('a' * (MAX_LENGTH_SHORT_REPR + 1)) is False
    lit_str = repr_utils.LiteralStr('test')
    assert has_short_repr(lit_str) is False  # Should be False for LiteralStr
    assert has_short_repr([]) is False  # False for other Sized objects
    assert has_short_repr(34) is True  # True for not Sized object


def test_literal_string():
    """Test YAML representers for correct serialization."""
    str_value = 'test'
    lit_str = repr_utils.LiteralStr(str_value)
    yaml_output = yaml.dump(lit_str)
    assert yaml_output != yaml.dump(str_value)
    assert yaml_output == yaml.dump(str_value, default_style='|')


def test_str_with_ts():
    str_value = 'test'
    str_with_ts = repr_utils.StrWithTS(str_value)
    yaml_output = yaml.dump(str_with_ts)
    # str_with_ts format should include the ts.
    assert yaml_output == yaml.dump(repr(str_with_ts).strip("'"))


def test_short_sequence():
    """Test sequence representation logic."""
    short_seq = ['a', 'b']
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
    omitted = repr_utils.Omitted(5)
    yaml_string = yaml.dump(omitted)
    assert yaml_string == '!Omitted\nomitted_elements: 5\n'


def test_represent_unknown_omitted():
    omitted = repr_utils.Omitted()
    yaml_string = yaml.dump(omitted, Dumper=yaml.Dumper)
    assert yaml_string == '!Omitted\nomitted_elements: .nan\n'


def test_represent_list_with_omitted():
    yaml_string = yaml.dump([2, repr_utils.Omitted(5), 3])
    assert yaml_string == "[2, !Omitted {omitted_elements: 5}, 3]\n"


@given(text(characters(codec='ascii', exclude_categories=['Cc', 'Cs'])))
def test_literal_str_yaml_representation(string):
    # pipe style incompatible with trailing spaces or empty strings
    stripped = string.strip()
    assume(stripped)
    literal = repr_utils.LiteralStr(stripped)
    yaml_literal = yaml.dump(literal)
    assert yaml_literal.startswith('|-\n')

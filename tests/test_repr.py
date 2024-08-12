from typing import Sequence

import pytest
import yaml  # typing: ignore
import pandas as pd
import torch
import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import text, characters, lists

from dry_torch.repr_utils import (
    LiteralStr,
    Omitted,
    recursive_repr,
    MAX_LENGTH_PLAIN_REPR,
    MAX_LENGTH_SHORT_REPR,
)

scalar_data = list(zip([1, -3.2, 1j, 'test_string', None]))
"""data for which recursive_repr is the identity function."""

list_data = [
    ([1, 2, 3], 3, [1, 2, 3]),
    ([1, 2, 3], 2, [1, Omitted(1), 3]),
    ([1, 2, 3, 4], 3, [1, Omitted(1), 3, 4]),
    ([1, 2, 3, 4], 2, [1, Omitted(2), 4]),
]
"""data limited in size by recursive_repr."""
dict_data = [
    ({1: 1, 2: 2, 3: 3}, 3, {'1': 1, '2': 2, '3': 3}),
    ({1: 1, 2: 2, 3: 3}, 2, {'1': 1, '2': 2, '...': Omitted(1)}),
    ({1: 1, 2: 2, 3: 3, 4: 4}, 2, {'1': 1, '2': 2, '...': Omitted(2)}),
]
"""dict limited in size by recursive_repr."""


@given(text(characters(codec='ascii', exclude_categories=['Cc', 'Cs'])))
def test_literal_str_yaml_representation(string):
    # pipe style incompatible with trailing spaces or empty strings
    stripped = string.strip()
    assume(stripped)
    literal = LiteralStr(stripped)
    yaml_literal = yaml.dump(literal, default_flow_style=False)
    assert yaml_literal.startswith('|-\n')


@given(
    lists(elements=text(), max_size=MAX_LENGTH_PLAIN_REPR)
)
def test_represent_short_sequence(sequence):
    yaml_string = yaml.dump(sequence, default_flow_style=False)
    assert yaml_string == yaml.dump(sequence, default_flow_style=True)


@given(
    lists(elements=text(max_size=MAX_LENGTH_SHORT_REPR),
          min_size=MAX_LENGTH_PLAIN_REPR + 1)
)
def test_represent_long_sequence(sequence):
    yaml_string = yaml.dump(sequence, default_flow_style=False)
    assert yaml_string == yaml.dump(sequence, default_flow_style=False)


@given(
    lists(elements=text(max_size=MAX_LENGTH_SHORT_REPR),
          min_size=1,
          max_size=MAX_LENGTH_PLAIN_REPR)
)
def test_represent_sequence_with_literal(sequence):
    sequence[0] = LiteralStr(sequence[0])
    yaml_string = yaml.dump(sequence, default_flow_style=False)
    assert yaml_string == yaml.dump(sequence, default_flow_style=False)


@given(
    lists(elements=text(min_size=MAX_LENGTH_SHORT_REPR),
          min_size=1,
          max_size=MAX_LENGTH_PLAIN_REPR)
)
def test_represent_sequence_with_long_strings(sequence):
    yaml_string = yaml.dump(sequence, default_flow_style=False)
    assert yaml_string == yaml.dump(sequence, default_flow_style=False)


def test_represent_omitted():
    omitted = Omitted(5)
    yaml_string = yaml.dump(omitted)
    assert yaml_string == '!Omitted\nomitted_elements: 5\n'


def test_represent_unknown_omitted():
    omitted = Omitted()
    yaml_string = yaml.dump(omitted, Dumper=yaml.Dumper)
    assert yaml_string == '!Omitted\nomitted_elements: .nan\n'


def test_represent_list_with_omitted():
    yaml_string = yaml.dump([2, Omitted(5), 3], default_flow_style=False)
    assert yaml_string == "[2, !Omitted {omitted_elements: 5}, 3]\n"


@pytest.mark.parametrize(['obj'], scalar_data)
def test_recursive_repr_simple_objects(obj: object):
    assert recursive_repr(obj) == obj


@pytest.mark.parametrize(['obj', 'max_size', 'expected'], list_data)
def test_recursive_repr_list(obj: Sequence, max_size: int, expected: Sequence):
    assert recursive_repr(obj, max_size=max_size) == expected


@pytest.mark.parametrize(['obj', 'max_size', 'expected'], list_data)
def test_recursive_repr_tuple(obj: Sequence, max_size: int, expected: Sequence):
    assert recursive_repr(tuple(obj), max_size=max_size) == tuple(expected)


@pytest.mark.parametrize(['obj', 'max_size', 'expected'], list_data)
def test_recursive_repr_set(obj: Sequence, max_size: int, expected: Sequence):
    assert recursive_repr(set(obj), max_size=max_size) == set(expected)


@pytest.mark.parametrize(['obj', 'max_size', 'expected'], dict_data)
def test_recursive_repr_dict(obj: Sequence, max_size: int, expected: Sequence):
    assert recursive_repr(obj, max_size=max_size) == expected


def test_recursive_repr_tensor():
    tensor = torch.tensor([[1.23456, 2.34567], [3.45678, 4.56789]])
    result = recursive_repr(tensor)
    assert isinstance(result, LiteralStr)


def test_recursive_repr_pandas():
    df = pd.DataFrame({'A': range(15), 'B': range(15)})
    result = recursive_repr(df, max_size=5)
    assert isinstance(result, LiteralStr)
    assert "A" in result
    assert "B" in result


def test_recursive_repr_numpy():
    array = np.array([[1.23456, 2.34567], [3.45678, 4.56789]])
    result = recursive_repr(array)
    assert isinstance(result, LiteralStr)
    assert "1.23" in result


def test_recursive_repr_class():
    class TestClass:
        def __init__(self):
            self.attr1 = 1
            self.attr2 = "string"

    obj = TestClass()
    result = recursive_repr(obj)
    assert 'TestClass' in result
    assert 'attr1' in result
    assert 'attr2' in result
    assert result['object'] == 'TestClass'


def test_yaml_dump():
    data = {
        'string': "This is a test string",
        'list': [1, 2, 3],
        'tensor': torch.tensor([1.0, 2.0, 3.0]),
        'dataframe': pd.DataFrame({'A': range(3)})
    }
    yaml_string = yaml.dump(data, Dumper=yaml.Dumper)
    assert "This is a test string" in yaml_string
    assert "- 1\n- 2\n- 3" in yaml_string
    assert "|-\n  [1.0, 2.0, 3.0]" in yaml_string
    assert "A:\n- 0\n- 1\n- 2" in yaml_string

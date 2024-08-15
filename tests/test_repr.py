from typing import Any

import pytest
import yaml  # type: ignore
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


class SimpleClass:
    int_value = 1
    string_value = 'text'


class LongClass(SimpleClass):
    long_string_value = 5 * [1, ]


class SlottedClass(SimpleClass):
    __slots__ = ('int_value', 'string_value')


scalar_data = [(elem, 0, elem) for elem in [1, -3.2, 1j, 'test_string', None]]
"""data for which recursive_repr is the identity function."""

list_data: list[tuple[list[int], int, list[int | Omitted]]] = [
    ([1, 2, 3], 3, [1, 2, 3]),
    ([1, 2, 3], 2, [1, Omitted(1), 3]),
    ([1, 2, 3, 4], 3, [1, Omitted(1), 3, 4]),
    ([1, 2, 3, 4], 2, [1, Omitted(2), 4]),
]

tuple_data = [(tuple(obj), max_length, tuple(expected))
              for obj, max_length, expected in list_data]

set_data = [(set(obj), max_length, set(expected))
            for obj, max_length, expected in list_data]

dict_data = [
    ({1: 1, 2: 2, 3: 3}, 3, {'1': 1, '2': 2, '3': 3}),
    ({1: 1, 2: 2, 3: 3}, 2, {'1': 1, '2': 2, '...': Omitted(1)}),
    ({1: 1, 2: 2, 3: 3, 4: 4}, 2, {'1': 1, '2': 2, '...': Omitted(2)}),
]

external_data = [
    (np.float32(1), 0, 1.),
    (np.array([]), 2, LiteralStr(np.array([]))),
    (np.array([1, 2, 3]), 2, LiteralStr('[1 ... 3]')),
    (torch.FloatTensor([1, 2, 3]), 2, LiteralStr('[1. ... 3.]')),
    (pd.DataFrame({'A': range(5), 'B': range(5)}),
     2,
     LiteralStr(
         '    A  B\n0   0  0\n.. .. ..\n4   4  4\n\n[5 rows x 2 columns]')
     ),
]

class_data = [
    (SimpleClass(), 2, 'SimpleClass')
]

repr_data: list[tuple[Any, int, Any]] = sum(
    (scalar_data, list_data, tuple_data, set_data, dict_data, external_data,
     class_data),
    []
)


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


@pytest.mark.parametrize(['obj', 'max_size', 'expected'], repr_data)
def test_recursive_repr_list(obj: object, max_size: int, expected: object):
    assert recursive_repr(obj, max_size=max_size) == expected


def test_recursive_repr_tensor():
    tensor = torch.tensor([[1.23456, 2.34567], [3.45678, 4.56789]])
    result = recursive_repr(tensor)
    assert isinstance(result, LiteralStr)

"""Tests for the "repr_utils" module."""

import dataclasses
import datetime

from collections.abc import Generator
from typing import NoReturn

import numpy as np
import torch

import pytest

import drytorch.utils.repr_utils

from drytorch.utils.repr_utils import (
    MAX_REPR_SIZE as ORIG_MAX_REPR,
)
from drytorch.utils.repr_utils import (
    CreatedAtMixin,
    DefaultName,
    LiteralStr,
    Omitted,
    _has_own_repr,
    _limit_size,
    recursive_repr,
)


class _NamedClass:
    name = DefaultName()

    def __init__(self) -> None:
        self.name = ''


class _NamedSubClass(_NamedClass):
    def __init__(self) -> None:
        super().__init__()


class _ClassWithAttributes:
    """Class with an attribute."""

    def __init__(self, int_value: int = 1) -> None:
        super().__init__()
        self.int_value = int_value
        self.string_value = 'text'
        return


class _ClassWithProperty:
    """Class with no attributes but a working and a not-implemented property."""

    int_value = 1
    string_value = 'text'

    @property
    def int_property(self) -> int:
        """Property that returns an int."""
        return 2

    @property
    def fail_property(self) -> NoReturn:
        """Property that raises an error."""
        raise ValueError('Failed')


@dataclasses.dataclass(slots=True)
class _SlottedClass:
    int_value: int = 1
    string_value: str = 'text'


@pytest.fixture
def _simple_class_with_property() -> Generator[_ClassWithProperty, None, None]:
    drytorch.utils.repr_utils.INCLUDE_PROPERTIES = True
    yield _ClassWithProperty()

    drytorch.utils.repr_utils.INCLUDE_PROPERTIES = False
    return


class TestCreatedAtMixin:
    """Test formatting as expected."""

    @pytest.fixture
    def created_at(self) -> CreatedAtMixin:
        """Set up the instance with no version."""
        return CreatedAtMixin()

    def test_property(self, created_at) -> None:
        """Test that the version is not an empty string."""
        assert isinstance(created_at.created_at, datetime.datetime)
        assert created_at.created_at_str


class TestDefaultName:
    """Test DefaultName class generates incremental names."""

    @pytest.fixture
    def class_instance(self) -> _NamedClass:
        """Set up a class containing the descriptor."""
        return _NamedClass()

    @pytest.fixture
    def other_instance(self) -> _NamedClass:
        """Set up a second class containing the descriptor."""
        return _NamedClass()

    @pytest.fixture
    def sub_class_instance(self) -> _NamedSubClass:
        """Set up a subclass containing the descriptor."""
        return _NamedSubClass()

    def test_default_name(
        self, class_instance, other_instance, sub_class_instance
    ) -> None:
        """Test DefaultName class generates incremental names."""
        assert class_instance.name == class_instance.__class__.__name__
        assert other_instance.name == f'{class_instance.__class__.__name__}_1'
        assert sub_class_instance.name == sub_class_instance.__class__.__name__


def get_atomic_data() -> list[
    tuple[int | str | complex | None, int, int | complex | str | None]
]:
    """Get data that should remain unchanged in recursive representation."""
    return [(elem, 0, elem) for elem in [1, -3.2, 1j, 'test_string', None]]


def get_list_data() -> list[tuple[list[int], int, list[int | Omitted]]]:
    """Get for lists with various sizes."""
    return [
        ([1, 2, 3], 3, [1, 2, 3]),
        ([1, 2, 3], 2, [1, Omitted(1), 3]),
        ([1, 2, 3, 4], 3, [1, Omitted(1), 3, 4]),
        ([1, 2, 3, 4], 2, [1, Omitted(2), 4]),
    ]


def get_tuple_data(
    list_data,
) -> list[tuple[tuple[int, ...], int, tuple[int | Omitted, ...]]]:
    """Get data for tuples with various sizes."""
    return [
        (tuple(obj), max_len, tuple(expected))
        for obj, max_len, expected in list_data
    ]


def get_set_data(list_data) -> list[tuple[set[int], int, set[int | Omitted]]]:
    """Get data for sets with various sizes."""
    return [
        (set(obj), max_len, set(expected))
        for obj, max_len, expected in list_data
    ]


def get_dict_data() -> list[
    tuple[dict[int, int], int, dict[str, int | Omitted]]
]:
    """Get data for dicts with various sizes."""
    return [
        ({1: 1, 2: 2, 3: 3}, 3, {'1': 1, '2': 2, '3': 3}),
        ({1: 1, 2: 2, 3: 3}, 2, {'1': 1, '2': 2, '...': Omitted(1)}),
    ]


def get_numpy_and_torch_data() -> list[
    tuple[int | float | np.ndarray | torch.Tensor, int, LiteralStr]
]:
    """External data with numpy arrays, torch tensors, and pandas DataFrames."""
    size_str = LiteralStr('Array of size (3,)\n')
    return [
        (np.float32(1), 0, 1.0),
        (np.array([1, 2, 3]), 2, size_str + LiteralStr('[1 ... 3]')),
        (torch.FloatTensor([1, 2, 3]), 2, size_str + LiteralStr('[1. ... 3.]')),
    ]


def get_class_data_with_attribute() -> list[
    tuple[_ClassWithAttributes, int, dict[str, int | str]]
]:
    """Get data for a class with no attributes."""
    data_class_expected = {
        'class': '_ClassWithAttributes',
        'int_value': 1,
        'string_value': 'text',
    }
    return [(_ClassWithAttributes(), 2, data_class_expected)]


def get_class_data_with_property() -> list[tuple[_ClassWithProperty, int, str]]:
    """Get data for a class with no attributes."""
    return [(_ClassWithProperty(), 2, '_ClassWithProperty')]


def get_data_class() -> list[tuple[_SlottedClass, int, dict[str, int | str]]]:
    """Get data for a class with slotted attributes."""
    data_class_expected = {
        'class': '_SlottedClass',
        'int_value': 1,
        'string_value': 'text',
    }
    return [(_SlottedClass(), 2, data_class_expected)]


def get_repr_data():
    """Return all data for recursive_repr tests."""
    list_data = get_list_data()
    return (
        get_atomic_data()
        + list_data
        + get_tuple_data(list_data)
        + get_set_data(list_data)
        + get_dict_data()
        + get_numpy_and_torch_data()
        + get_class_data_with_attribute()
        + get_class_data_with_property()
        + get_data_class()
    )


@pytest.mark.parametrize(['obj', 'max_size', 'expected'], get_repr_data())
def test_recursive_repr(obj: object, max_size: int, expected: object) -> None:
    """Test the recursive_repr function with various data types."""
    drytorch.utils.repr_utils.MAX_REPR_SIZE = max_size
    assert recursive_repr(obj) == expected


def test_property_repr(_simple_class_with_property) -> None:
    """Test the recursive_repr function with various data types."""
    assert recursive_repr(_simple_class_with_property) == {
        'class': '_ClassWithProperty',
        'int_property': 2,
        'fail_property': 'Failed',
    }


def test_limit_size() -> None:
    """Test limit_size function limits the size of iterable and adds Omitted."""
    drytorch.utils.repr_utils.MAX_REPR_SIZE = ORIG_MAX_REPR
    long_list = list(range(20))
    result = _limit_size(long_list)
    assert len(result) == 11  # limited size
    assert isinstance(result[5], Omitted)
    assert result[:5] == [0, 1, 2, 3, 4]
    assert result[-5:] == [15, 16, 17, 18, 19]


def test_has_own_repr() -> None:
    """Test has_own_repr function to check if __repr__ has been overridden."""
    assert _has_own_repr(_NamedClass()) is False

    class _CustomReprClass:
        def __repr__(self):
            return 'Custom Representation'

    assert _has_own_repr(_CustomReprClass()) is True


def test_pandas_print_options() -> None:
    """Test PandasPrintOptions context manager changes Pandas settings."""
    pd = pytest.importorskip('pandas')
    drytorch.utils.repr_utils.MAX_REPR_SIZE = ORIG_MAX_REPR
    original_max_rows = pd.get_option('display.max_rows')
    original_max_columns = pd.get_option('display.max_columns')
    df = pd.DataFrame({'A': range(5), 'B': range(5)})
    expected_df_repr = LiteralStr(
        '    A  B\n0   0  0\n.. .. ..\n4   4  4\n\n[5 rows x 2 columns]'
    )
    drytorch.utils.repr_utils.MAX_REPR_SIZE = 2

    assert recursive_repr(df) == expected_df_repr
    assert pd.get_option('display.max_rows') == original_max_rows
    assert pd.get_option('display.max_columns') == original_max_columns

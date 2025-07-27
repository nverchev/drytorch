"""Tests for the "repr_utils" module."""

import numpy as np
import torch

import pytest

from drytorch.utils.repr_utils import (
    DefaultName,
    LiteralStr,
    Omitted,
    Versioned,
    has_own_repr,
    limit_size,
    recursive_repr,
)


# Test class for DefaultName
class _NamedClass:
    name = DefaultName()

    def __init__(self) -> None:
        self.name = ''


class _NamedSubClass(_NamedClass):
    def __init__(self) -> None:
        super().__init__()


# Test classes for recursive representation
class _SimpleClass:
    int_value = 1
    string_value = 'text'


class _LongClass(_SimpleClass):
    long_string_value = 5 * [1, ]


class _SlottedClass(_SimpleClass):
    __slots__ = ('int_value', 'string_value')


class TestVersioned:
    """Test formatting as expected."""

    @pytest.fixture
    def versioned(self) -> Versioned:
        """Set up the instance."""
        return Versioned()

    def test_property(self, versioned):
        """Test that the version is not an empty string."""
        assert versioned.created_at


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

    def test_default_name(self,
                          class_instance,
                          other_instance,
                          sub_class_instance) -> None:
        """Test DefaultName class generates incremental names."""
        assert class_instance.name == class_instance.__class__.__name__
        assert other_instance.name == f'{class_instance.__class__.__name__}_1'
        assert sub_class_instance.name == sub_class_instance.__class__.__name__


# Scalar data that should remain unchanged in recursive representation
scalar_data = [(elem, 0, elem) for elem in [1, -3.2, 1j, 'test_string', None]]

# Test data for list, tuple, set, and dict types with various sizes
list_data: list[tuple[list[int], int, list[int | Omitted]]]

list_data = [
    ([1, 2, 3], 3, [1, 2, 3]),
    ([1, 2, 3], 2, [1, Omitted(1), 3]),
    ([1, 2, 3, 4], 3, [1, Omitted(1), 3, 4]),
    ([1, 2, 3, 4], 2, [1, Omitted(2), 4]),
]
tuple_data: list[tuple[tuple[int, ...], int, tuple[int | Omitted, ...]]]
tuple_data = [(tuple(obj), max_len, tuple(expected))
              for obj, max_len, expected in list_data]
set_data: list[tuple[set[int], int, set[int | Omitted]]]
set_data = [(set(obj), max_len, set(expected))
            for obj, max_len, expected in list_data]
dict_data = [
    ({1: 1, 2: 2, 3: 3}, 3, {'1': 1, '2': 2, '3': 3}),
    ({1: 1, 2: 2, 3: 3}, 2, {'1': 1, '2': 2, '...': Omitted(1)}),
]

# External data using numpy arrays, torch tensors, and pandas DataFrames
numpy_and_torch_data = [
    (np.float32(1), 0, 1.),
    (np.array([1, 2, 3]), 2, LiteralStr('[1 ... 3]')),
    (torch.FloatTensor([1, 2, 3]), 2, LiteralStr('[1. ... 3.]')),
]

class_data = [
    (_SimpleClass(), 2, '_SimpleClass')
]

repr_data = (scalar_data +
             list_data +
             tuple_data +
             set_data +
             dict_data +
             numpy_and_torch_data +
             class_data)


@pytest.mark.parametrize(['obj', 'max_size', 'expected'], repr_data)
def test_recursive_repr(obj: object, max_size: int, expected: object) -> None:
    """Test the recursive_repr function with various data types."""
    assert recursive_repr(obj, max_size=max_size) == expected


def test_limit_size() -> None:
    """Test limit_size function limits the size of iterable and adds Omitted."""
    long_list = list(range(20))
    result = limit_size(long_list, max_size=10)
    assert len(result) == 11  # Includes Omitted
    assert isinstance(result[5], Omitted)  # Middle item should be Omitted
    assert result[:5] == [0, 1, 2, 3, 4]  # Start is intact
    assert result[-5:] == [15, 16, 17, 18, 19]  # End is intact


def test_has_own_repr() -> None:
    """Test has_own_repr function to check if __repr__ has been overridden."""
    assert has_own_repr(_NamedClass()) is False

    class _CustomReprClass:
        def __repr__(self):
            return 'Custom Representation'

    assert has_own_repr(_CustomReprClass()) is True


def test_pandas_print_options() -> None:
    """Test PandasPrintOptions context manager changes Pandas settings."""
    pd = pytest.importorskip('pandas')
    original_max_rows = pd.get_option('display.max_rows')
    original_max_columns = pd.get_option('display.max_columns')
    df = pd.DataFrame({'A': range(5), 'B': range(5)})
    expected_df_repr = LiteralStr(
        '    A  B\n0   0  0\n.. .. ..\n4   4  4\n\n[5 rows x 2 columns]'
    )
    assert recursive_repr(df, max_size=2) == expected_df_repr

    # After context, original values should be restored
    assert pd.get_option('display.max_rows') == original_max_rows
    assert pd.get_option('display.max_columns') == original_max_columns

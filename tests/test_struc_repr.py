import pytest
from typing import Iterable, Iterator, Self

import numpy as np
import pandas as pd
import torch
import yaml  # type: ignore

from dry_torch.repr_utils import PandasPrintOptions
from dry_torch.repr_utils import recursive_repr


def get_example_dataframe() -> tuple[pd.DataFrame, str]:
    df = pd.DataFrame({'col_' + str(i): 10 * [3.141592] for i in range(10)})
    expected = ('    col_0  ...  col_9\n'
                '0   3.142  ...  3.142\n'
                '..    ...  ...    ...\n'
                '9   3.142  ...  3.142\n\n'
                '[10 rows x 10 columns]')
    return df, expected


# class with slots
class OneSlot:
    __slots__ = ['single_slot']

    def __init__(self):
        self.single_slot = 'only value'

    # class to test dict_repr


class ComplexClass:
    # class attributes are not represented
    max_size = 2
    df, expected_df = get_example_dataframe()

    def __init__(self):
        # discard private attribute when documenting the object
        self._private_attr = None

        # want to keep numeric values as numeric values
        self.zero_attr = 0

        # keep strings as they are
        self.lorem = "Lorem ipsum dolor sit amet"

        # transform tensors and arrays into strings
        self.torch_tensor = np.pi * torch.ones(4, 4)

        self.df = self.df

        # this will be longer than what we are going to allow
        long_generator = range(1, 4)
        self.complex_struc = {
            'tuple': tuple(long_generator),
            'set': set(long_generator),
            'list': list(long_generator),
        }
        # write the model_name of the function
        self.fun = lambda: None

        # object with slots
        self.object_with_one_slot = OneSlot()

    # expected representation of the object for documentation
    def expected_struc_repr(self):
        expected = {
            'class': self.__class__.__name__,
            'zero_attr': self.zero_attr,
            'lorem': self.lorem,
            'complex_struc': {'tuple': (1, '...', 3),
                              '...': None,
                              'list': [1, '...', 3]},
            'torch_tensor': '[[3.142 ... 3.142]\n ...\n [3.142 ... 3.142]]',
            'fun': '<lambda>',
            'object_with_one_slot': {'class': 'OneSlot',
                                     'single_slot': 'only value'},
            'df': self.expected_df
        }
        return expected


def test_pandas_print_options() -> None:
    df, expected = get_example_dataframe()
    with PandasPrintOptions(precision=3, max_rows=2, max_columns=2):
        assert str(rf'{df}') == expected


def test_struc_repr() -> None:
    complex_instance = ComplexClass()
    out = recursive_repr(complex_instance, max_size=complex_instance.max_size)
    print(out)
    assert out == complex_instance.expected_struc_repr()
    with open('test.yml', 'w') as yaml_file:
        yaml.dump(out, yaml_file, sort_keys=False)


def test_break_recursion() -> None:
    # struc_repr should be safe from simple recursive objects
    class ContainSelf:

        def __init__(self):
            self.self = self

    _ = recursive_repr(ContainSelf())

    # this should lead to recursion error
    class BreakRecursive(Iterable):

        def __iter__(self) -> Iterator[tuple[Self]]:
            return zip(self)

    with pytest.raises(RecursionError):
        recursive_repr(BreakRecursive())

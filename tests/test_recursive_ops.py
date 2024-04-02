import pytest
from typing import Iterator, Iterable, Self, reveal_type
import numpy as np
import torch
from custom_trainer.recursive_ops import recursive_apply, struc_repr


def test_recursive_apply() -> None:
    expected_type = float
    tuple_data = (1., [1, 2])
    dict_data = {'list': tuple_data}

    # fail because it expects floats and not int
    with pytest.raises(TypeError):
        recursive_apply(struc=dict_data, expected_type=expected_type, func=lambda x: 2 * x)

    def str_torch(x: torch.Tensor) -> float:
        return x.item()

    # change int into floats
    new_tuple_data = (torch.tensor(1.), [torch.tensor(1.), torch.tensor(2.)])
    new_dict_data = {'list': new_tuple_data}
    out = recursive_apply(struc=new_dict_data, expected_type=torch.Tensor, func=str_torch)
    a = list(out.values())[0][1]
    out2 = recursive_apply(struc=1, expected_type=int, func=str)
    reveal_type(out2)

    assert out == {'list': [2., (2., 4.)]}


def test_struc_repr() -> None:
    # class with slots
    class OneSlot:
        __slots__ = ['single_slot']

        def __init__(self):
            self.single_slot = 'only value'

    # class to test dict_repr
    class ComplexClass:
        # class attributes are not represented
        max_length = 2

        def __init__(self):
            # discard these attribute when documenting the object
            self.null_attr = None
            self.empty_set = set()
            self.empty_list = []
            self.empty_dict = {}
            self.empty_str = ''

            # want to keep this
            self.zero_attr = 0

            # keep strings as they are
            self.lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit"

            # transform tensors and arrays into strings
            self.torch_tensor = np.pi * torch.ones(4, 4)

            # this will be longer than what we are going to allow
            long_generator = range(1, 4)
            self.complex_struct = {'tuple': tuple(long_generator),
                                   'list': list(long_generator),
                                   'set': set(long_generator)}

            # write the name of the function
            self.fun = int

            # object with slots
            self.object_with_one_slot = OneSlot()

        # expected representation of the object for documentation
        def expected_dict_repr(self):
            expected = {'class': self.__class__.__name__,
                        'zero_attr': self.zero_attr,
                        'lorem': self.lorem,
                        'complex_struct': {'tuple': (1, '...', 3),
                                           '...': '',
                                           'set': {1, '...', 3}},
                        'torch_tensor': '[[3.142 ... 3.142]\n ...\n [3.142 ... 3.142]]',
                        'fun': self.fun.__name__,
                        'object_with_one_slot': {'class': 'OneSlot', 'single_slot': 'only value'}
                        }
            return expected

    complex_instance = ComplexClass()
    out = struc_repr(complex_instance, max_length=complex_instance.max_length)
    assert out == complex_instance.expected_dict_repr()


def test_break_recursive() -> None:
    # struc_repr should be protected from simple recursive objects
    class ContainSelf:

        def __init__(self):
            self.self = self

    struc_repr(ContainSelf())

    # should lead to recursion error
    class BreakRecursive(Iterable):

        def __iter__(self) -> Iterator[tuple[Self]]:
            return zip(self)

    with pytest.raises(RecursionError):
        struc_repr(BreakRecursive())

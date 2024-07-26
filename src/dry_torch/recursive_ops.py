"""This module contains functions for nested containers."""

from typing import Callable, Type, TypeVar, overload

import torch

from dry_torch import exceptions
from dry_torch import protocols as p

_T = TypeVar('_T')
_C = TypeVar('_C', dict, list, set, tuple, p.NamedTupleProtocol)


@overload
def recursive_apply(
        obj: _T,
        expected_type: Type[_T],
        func: Callable[[_T], _T]
) -> _T:
    ...


@overload
def recursive_apply(
        obj: _C,
        expected_type: Type[_T],
        func: Callable[[_T], _T]
) -> _C:
    ...


def recursive_apply(obj, expected_type, func):
    """
    Function that looks for an expected type and applies a given function.

    It looks inside the items of lists, tuples and sets and the values of
    dictionaries and returns the built-in container. Subtypes are not supported,
    except for NamedTuples, for which a new instance with the modified objects
    is returned.

    Args:
        obj: a dictionary, list, set, tuple, NamedTuples or the expected object.
        expected_type: the type of the expected objects contained in obj.
        func: a function that modifies objects of the expected type.

    Returns:
        The modified object or a copy of obj containing the modified objects.

    Raises:
        TypeError: if obj is un unexpected object.
    """
    obj_class = type(obj)

    if obj_class == expected_type:
        return func(obj)

    if obj_class == dict:
        return {
            k: recursive_apply(v, expected_type, func) for k, v in obj.items()
        }

    if isinstance(obj, p.NamedTupleProtocol):
        # noinspection PyProtectedMember
        return obj._make(
            recursive_apply(item, expected_type, func) for item in obj
        )

    if obj_class in (list, set, tuple):
        return obj_class(
            recursive_apply(item, expected_type, func) for item in obj
        )

    raise exceptions.FuncNotApplicableError(func, obj_class)


def recursive_to(container: _C, device: torch.device) -> _C:
    """
    Function that changes the device of tensors inside a container.

    Args:
        container: a Tensor container made of dict, list, set or tuples.
        device: the target device.

    Returns:
        the same container with the tensor on the target device.
    """

    def to_device(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device)

    return recursive_apply(container,
                           expected_type=torch.Tensor,
                           func=to_device)

from typing import Any, Callable, Type, overload, TypeVar
import numpy as np
import pandas as pd
import torch

from dry_torch import exceptions
from dry_torch import protocols as p

_T = TypeVar('_T')
_C = TypeVar('_C', dict, list, set, tuple)
_Array = TypeVar('_Array', torch.Tensor, np.ndarray, pd.DataFrame, pd.Series)


@overload
def recursive_apply(
        obj: _Array,
        expected_type: Type[_Array],
        func: Callable[[_Array], _T]
) -> _T:
    ...


@overload
def recursive_apply(
        obj: _C,
        expected_type: Type[_Array],
        func: Callable[[_Array], _Array]
) -> _C:
    ...


@overload
def recursive_apply(
        obj: _C,
        expected_type: Type[_Array],
        func: Callable[[_Array], _T]
) -> Any:
    ...  # mypy do not support TypeVar with higher kinds


def recursive_apply(obj, expected_type, func):
    """
    Function that looks for an expected type and applies a given function.

    It looks inside the items of lists, tuples and sets and the values of
    dictionaries and returns the built-in container. Subtypes are not supported,
     except for NamedTuples, which are returned as they are.

    Args:
        obj: container made of dictionaries, list, sets, tuples and NamedTuples.
        expected_type: the type of the objects contained in obj.
        func: the function to apply to objects of the expected type.

    Returns:
        Any: a copy of the structure in the argument with the modified objects.

    Raises:
        TypeError: if obj stores un unexpected object.
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

    raise exceptions.FuncNotApplicableError(func, obj.__class__)


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

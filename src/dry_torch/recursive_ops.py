from typing import Any, Callable, Type, overload, TypeVar
import numpy as np
import pandas as pd
import torch

_T = TypeVar('_T')
_C = TypeVar('_C', dict, list, set, tuple)
_Array = TypeVar('_Array', torch.Tensor, np.ndarray, pd.DataFrame, pd.Series)


@overload
def recursive_apply(
        struc: _Array,
        expected_type: Type[_Array],
        func: Callable[[_Array], _T]
) -> _T:
    ...


@overload
def recursive_apply(
        struc: _C,
        expected_type: Type[_Array],
        func: Callable[[_Array], _Array]
) -> _C:
    ...


@overload
def recursive_apply(
        struc: _C,
        expected_type: Type[_Array],
        func: Callable[[_Array], _T]
) -> Any:
    ...  # mypy do not support TypeVar with higher kinds


def recursive_apply(struc, expected_type, func):
    """
    It recursively looks for type T in dictionary, lists, tuples and sets and
    applies func.
    It can be used for changing device in structured (nested) formats.

    Args:
        struc: an object that stores or is an object of type T.
        Can only contain dictionaries, list, sets, tuples or T.
        expected_type: the type T that struc stores.
        func: the function that we want to apply to objects of type T.

    Returns:
        Any: a copy of the structure in the argument with the modified objects.

    Raises:
        TypeError: if struc stores objects of a different type than T, dict,
        list, set or tuple.
    """
    if isinstance(struc, expected_type):
        return func(struc)
    if isinstance(struc, dict):
        return {
            k: recursive_apply(v, expected_type, func) for k, v in struc.items()
        }
    if isinstance(struc, (list, set, tuple)):
        return type(struc)(
            recursive_apply(item, expected_type, func) for item in struc
        )
    if isinstance(struc, expected_type):
        return func(struc)
    raise TypeError(f' Cannot apply {func} on Datatype {type(struc).__name__}')


def recursive_to(container: _C, device: torch.device) -> _C:
    """
    Change the devices of tensor inside a container.

    Args:
        container: a container made of dict, list, set or tuples that contains
        Tensors.
        device: the target device
    """

    def to_device(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device)

    return recursive_apply(container,
                           expected_type=torch.Tensor,
                           func=to_device)

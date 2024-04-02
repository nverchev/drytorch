from typing import Any, TypeVar, Callable, Type, overload
import numpy as np
import pandas as pd
import torch
from .context_managers import PandasPrintOptions

BaseArray = torch.Tensor | pd.DataFrame | pd.Series | np.ndarray | pd.Index
Atomic = TypeVar('Atomic', int, float, complex, str)
Array = TypeVar('Array', int, float, torch.Tensor, pd.DataFrame, pd.Series, np.ndarray, pd.Index)
T = TypeVar('T')
C = TypeVar('C', dict, list, set, tuple)


@overload
def recursive_apply(struc: Array, expected_type: Type[Array], func: Callable[[Array], T]) -> T:
    ...


@overload
def recursive_apply(struc: C, expected_type: Type[Array], func: Callable[[Array], Array]) -> C:
    ...


@overload
def recursive_apply(struc: C, expected_type: Type[Array], func: Callable[[Array], T]) -> Any:
    ...  # mypy do not support TypeVar with higher kinds


def recursive_apply(struc, expected_type, func):
    """
    It recursively looks for type T in dictionary, lists, tuples and sets and applies func.
    It can be used for changing device in structured (nested) formats.

    Args:
        struc: an object that stores or is an object of type T. Can only contain dictionaries, list, sets, tuples or T.
        expected_type: the type T that struc stores.
        func: the function that we want to apply to objects of type T.

    Returns:
        Any: a copy of the structure in the argument with the modified objects.

    Raises:
        TypeError: if struc stores objects of a different type than T, dict, list, set or tuple.
    """
    if isinstance(struc, expected_type):
        return func(struc)
    if isinstance(struc, dict):
        return {k: recursive_apply(v, expected_type, func) for (k, v) in struc.items()}
    if isinstance(struc, (list, set, tuple)):
        return type(struc)(recursive_apply(item, expected_type, func) for item in struc)
    if isinstance(struc, expected_type):
        return func(struc)
    raise TypeError(f' Cannot apply {func} on Datatype {type(struc).__name__}')


@overload
def struc_repr(struc: Atomic, *, max_length: int = ...) -> Atomic:
    ...


@overload
def struc_repr(struc: Array | Type, *, max_length: int = ...) -> str:
    ...


@overload
def struc_repr(struc: Any, *, max_length: int = ...) -> Any:
    ...


def struc_repr(struc, *, max_length=10):
    """
    It attempts full documentation of a complex object.
    It recursively represents each attribute or element of the object.
    To prevent excessive data from being included in the representation, it limits the size of containers.
    The readable representation contains only strings and numbers

    Args:
        struc: a complex object with that may contain other objects.
        max_length: limits the size of iterators and arrays.

    Returns:
        a readable representation of the object.
    """

    if isinstance(struc, (int, float, complex, str)):
        return struc

    if isinstance(struc, (type, type(lambda: ...))):  # no builtin variable for the function class!
        return struc.__name__

    if isinstance(struc, (pd.DataFrame, pd.Series, pd.Index)):
        with PandasPrintOptions(precision=3, max_rows=max_length, max_columns=max_length):
            return str(struc)

    if isinstance(struc, np.ndarray):
        with np.printoptions(precision=3, suppress=True, threshold=max_length, edgeitems=max_length // 2):
            return str(struc)

    if isinstance(struc, torch.Tensor):
        struc = struc.detach().cpu()

    if hasattr(struc, 'numpy'):
        try:
            return struc_repr(struc.numpy(), max_length=max_length)
        except TypeError as te:
            raise AttributeError('numpy should be a method without additional parameters') from te

    if hasattr(struc, '__iter__'):
        struc_list = list(struc)
        struc_list = struc_list[:max_length // 2] + ['...'] + struc_list[-max_length // 2:]
        struc_list = [struc_repr(item, max_length=max_length) for item in struc_list]
        if hasattr(struc, 'get'):  # assume that the iterable contains keys
            return {k: struc_repr(struc.get(k, ''), max_length=max_length) for k in struc_list}
        if isinstance(struc, (list, set, tuple)):
            return type(struc)(struc_list)
        return struc_list

    slots = getattr(struc, '__slots__', [])
    dict_attr = getattr(struc, '__dict__', {})
    if slots or dict_attr:
        dict_attr |= {name: getattr(struc, name) for name in slots}
        dict_attr = {k: struc_repr(v, max_length=max_length) for k, v in dict_attr.items()
                     if v not in ({}, [], set(), '', None, struc)}
        return {'class': type(struc).__name__} | dict_attr

    return repr(struc)

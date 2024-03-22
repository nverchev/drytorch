from typing import Any, TypeVar, Callable, Type, Collection
import numpy as np
import pandas as pd
import torch

import warnings

from .context_managers import PandasPrintOptions

T = TypeVar('T')
# nested collection that contains type T
C = Collection[T | 'C'] | T


class ConversionWarning(UserWarning):
    pass


def recursive_apply(struc: C, expected_type: Type[T], func: Callable[[T], Any]) -> C:
    """
    It recursively looks for type T in collection objects and applies func.
    It can be used for changing device in structured (nested) formats.

    Args:
        struc: a complex collection object that stores objects of type T
        expected_type: the type T that struc stores
        func: the function that we want to apply to objects of type T

    Returns:
        struc: a copy of the structure in the argument with the modified objects

    Raises:
        ConversionWarning: collection object cannot be instantiated in a standard way
        TypeError: struc stores non collection objects of a different type than T
    """
    collection = type(struc)
    if isinstance(struc, expected_type):
        return func(struc)
    if hasattr(struc, 'items'):
        try:
            return collection({k: recursive_apply(v, expected_type, func) for k, v in struc.items() if v is not struc})
        except TypeError:
            message = f'Collection object of type {collection.__name__} converted to dictionary.'
            warnings.warn(message, ConversionWarning)
            return {k: recursive_apply(v, expected_type, func) for k, v in struc.items()}
    if hasattr(struc, '__iter__'):
        tuple_struc = tuple(recursive_apply(item, expected_type, func) for item in struc if item is not struc)
        try:
            return collection(tuple_struc)
        except TypeError:
            message = f'Collection object of type {collection.__name__} converted to tuple.'
            warnings.warn(message, ConversionWarning)
            return tuple_struc
    raise TypeError(f' Cannot apply {func} on Datatype {collection.__name__}')


def recursive_repr(struc: Any, max_length: int = 10) -> Any:
    """
    It attempts full documentation of a complex object.
    It recursively represents each attribute or element of the object.
    To prevent excessive data from being included in the representation, it limits the size of containers.

    Args:
        struc: a complex object with that may contain other objects
        max_length: limits the size of iterators and arrays

    Returns:
        a readable representation of an object
    """

    if isinstance(struc, str):
        return struc

    if isinstance(struc, (type, type(lambda: ...))):  # no builtin variable for the function class!
        return struc.__name__

    if isinstance(struc, (pd.DataFrame, pd.Series)):
        with PandasPrintOptions(precision=3, max_rows=max_length, max_columns=max_length):
            return str(struc)

    if isinstance(struc, np.ndarray):
        with np.printoptions(precision=3, suppress=True, threshold=max_length, edgeitems=max_length // 2):
            return str(struc)

    if isinstance(struc, torch.Tensor):
        struc = struc.detach().cpu()

    if hasattr(struc, 'numpy'):
        try:
            return recursive_repr(struc.numpy(), max_length)
        except TypeError as te:
            raise AttributeError('numpy should be a method without additional parameters') from te

    if hasattr(struc, '__iter__'):
        iterable = type(struc)
        struc_list = list(struc)
        struc_list = struc_list[:max_length // 2] + ['...'] + struc_list[-max_length // 2:]
        struc_list = [recursive_repr(item, max_length) for item in struc_list if item is not struc]
        if hasattr(struc, 'get'):  # assume that the iterable contains keys
            dct = {k: recursive_repr(struc.get(k, ''), max_length) for k in struc_list}
            try:
                return iterable(dct)
            except TypeError:
                return dct
        try:
            return iterable(struc_list)
        except TypeError:
            return tuple(struc_list)

    slots = getattr(struc, '__slots__', [])
    dict_attr = getattr(struc, '__dict__', {})
    if slots or dict_attr:
        dict_attr |= {name: getattr(struc, name) for name in slots}
        dict_attr = {k: recursive_repr(v, max_length) for k, v in dict_attr.items()
                     if v not in ({}, [], set(), '', None, struc)}
        return {'class': type(struc).__name__} | dict_attr

    return struc

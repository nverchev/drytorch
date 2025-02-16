"""Utilities to extract readable documentation from any object."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import datetime
import functools
import itertools
import math
import numbers
import types
from typing import Any, Optional, cast

import numpy as np
import torch


class StrWithTS(str):
    """A string that adds a timestamp."""
    ts_fmt = '%Y-%m-%dT%H:%M:%S'
    len_ts = 20  # len('.2017-01-12T14:12:06')

    def __new__(cls, str_: str) -> StrWithTS:
        str_with_timestamp = f'{str_}.{datetime.datetime.now():{cls.ts_fmt}}'
        return cast(StrWithTS, super().__new__(cls, str_with_timestamp))

    def __format__(self, format_spec: str) -> str:
        if format_spec is None:
            return super().__format__('s')
        return self[:-self.len_ts].__format__('s')


class DefaultName:
    """Add a counter to a prefix"""

    def __init__(self):
        """
        """
        self._prefixes = dict[str, itertools.count]()

    def __get__(self, instance: Any, objtype: Optional[type] = None) -> str:
        """Return the default name for the instance or class."""
        return instance.__name

    def __set__(self, instance: Any, value: str) -> None:
        """Return the default name for the instance or class."""
        value = value if value else instance.__class__.__name__
        count_iter = self._prefixes.setdefault(value, itertools.count())
        if count_value := next(count_iter):
            value = f'{value}_{count_value}'
        instance.__name = StrWithTS(value)
        return


try:
    import pandas as pd

except (ImportError, ModuleNotFoundError):
    PandasObject = type(object())
    pd = types.ModuleType('Unreachable module.')

else:

    class PandasPrintOptions:
        """
        Context manager to temporarily set Pandas display options.

        Args:
            precision: Number of digits of precision for floating point output.
            max_rows: Maximum number of rows to display.
            max_columns: Maximum number of columns to display.
        """

        def __init__(self,
                     precision: int = 3,
                     max_rows: int = 10,
                     max_columns: int = 10) -> None:
            self.options: dict[str, int] = {
                'display.precision': precision,
                'display.max_rows': max_rows,
                'display.max_columns': max_columns,
            }
            self.original_options: dict[str, Any] = {}

        def __enter__(self) -> None:
            self.original_options.update(
                {key: pd.get_option(key) for key in self.options}
            )
            for key, value in self.options.items():
                pd.set_option(key, value)

        def __exit__(self,
                     exc_type: None = None,
                     exc_val: None = None,
                     exc_tb: None = None) -> None:
            for key, value in self.original_options.items():
                pd.set_option(key, value)


    PandasObject = pd.core.base.PandasObject


class LiteralStr(str):
    """YAML will attempt to use the pipe style for this class."""


@dataclasses.dataclass(frozen=True)
class Omitted:
    """Class for objects that represent omitted values in an iterable."""
    count: float = math.nan


def has_own_repr(obj: Any) -> bool:
    """Function that indicates whether __repr__ has been overridden."""
    return not obj.__repr__().endswith(str(hex(id(obj))) + '>')


def limit_size(container: Iterable, max_size: int) -> list:
    """Function that limits the size of iterables and adds an Omitted object."""
    # prevents infinite iterators
    if hasattr(container, '__len__'):
        listed = list(container)
        if len(listed) > max_size:
            omitted = [Omitted(len(listed) - max_size)]
            listed = listed[:max_size // 2] + omitted + listed[-max_size // 2:]
    else:
        listed = []
        iter_container = container.__iter__()
        for _ in range(max_size):
            try:
                value = next(iter_container)
                listed.append(value)
            except StopIteration:
                break
        else:
            listed.append([Omitted()])
    return listed


@functools.singledispatch
def recursive_repr(obj: object, *, max_size: int = 10) -> Any:
    """
    Function that attempts a hierarchical representation of a given object.

    It recursively represents each attribute of the object or the contained
    items in tuple, list, sets and dictionaries. The latter are limited in
    size by keeping max_size elements and replacing the others with an Omitted
    instance.

    Arrays are represented using pandas and numpy array representation. Numbers
    are return as they are or converted to python types.

    Args:
        obj: the object to represent.
        max_size: max length of iterators and arrays.

    Returns:
        a readable representation of the object.
    """
    class_name = obj.__class__.__name__
    dict_attr = getattr(obj, '__dict__', {})
    dict_attr |= {name: getattr(obj, name)
                  for name in getattr(obj, '__slots__', [])}
    dict_str: dict[str, Any] = {}
    for k, v in dict_attr.items():
        if k[0] == '_' or v is obj or v is None:
            continue
        if hasattr(v, '__len__'):
            if not v.__len__():
                continue
        dict_str[k] = recursive_repr(v, max_size=max_size)

    if dict_str:
        return {'class': class_name} | dict(sorted(dict_str.items()))

    return repr(obj) if has_own_repr(obj) else class_name


@recursive_repr.register
def _(obj: Omitted, *, max_size: int = 10) -> Omitted:
    _not_used = max_size
    return obj


@recursive_repr.register
def _(obj: str, *, max_size: int = 10) -> str:
    _not_used = max_size
    return obj


@recursive_repr.register
def _(obj: None, *, max_size: int = 10) -> None:
    _not_used = max_size
    return obj


@recursive_repr.register
def _(obj: numbers.Number, *, max_size: int = 10) -> numbers.Number:
    if hasattr(obj, 'item'):
        obj = obj.item()
    _not_used = max_size
    return obj


@recursive_repr.register
def _(obj: tuple, *, max_size: int = 10) -> tuple:
    return tuple(recursive_repr(item, max_size=max_size)
                 for item in limit_size(obj, max_size=max_size))


@recursive_repr.register
def _(obj: list, *, max_size: int = 10) -> list:
    return [recursive_repr(item, max_size=max_size)
            for item in limit_size(obj, max_size=max_size)]


@recursive_repr.register
def _(obj: set, *, max_size: int = 10) -> set:
    return {recursive_repr(item, max_size=max_size)
            for item in limit_size(obj, max_size=max_size)}


@recursive_repr.register
def _(obj: dict, *, max_size: int = 10) -> dict[str, Any]:
    out_dict: dict[str, Any] = {
        str(key): recursive_repr(value, max_size=max_size)
        for key, value in list(obj.items())[:max_size]}
    if len(obj) > max_size:
        out_dict['...'] = Omitted(len(obj) - max_size)
    return out_dict


@recursive_repr.register
def _(obj: torch.Tensor, *, max_size: int = 10) -> LiteralStr:
    _not_used = max_size
    return recursive_repr(obj.detach().cpu().numpy(), max_size=max_size)


@recursive_repr.register
def _(obj: PandasObject,  # type: ignore
      *,
      max_size: int = 10) -> LiteralStr:
    # only called when Pandas is imported
    with PandasPrintOptions(max_rows=max_size, max_columns=max_size):
        return LiteralStr(obj)


@recursive_repr.register
def _(obj: np.ndarray, *, max_size: int = 10) -> LiteralStr:
    size_factor = 2 ** (+obj.ndim - 1)
    with np.printoptions(precision=3,
                         suppress=True,
                         threshold=max_size // size_factor,
                         edgeitems=max_size // (size_factor * 2)):
        return LiteralStr(obj)


@recursive_repr.register(type)
def _(obj, *, max_size: int = 10) -> str:
    _not_used = max_size
    return obj.__name__


@recursive_repr.register(types.FunctionType)
def _(obj, *, max_size: int = 10) -> str:
    _not_used = max_size
    return obj.__name__

"""Utilities to extract readable documentation from any object."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import datetime
import functools
import math
import numbers
import types
from typing import Any, cast

import numpy as np
import torch


class DefaultName:
    """Add a counter to a prefix"""

    def __init__(self, start: int = -1):
        """
        Args:
            start: initial count value.
        """
        self.prefix = 'default'
        self.count_defaults = start

    def __get__(self, instance: Any, owner: type) -> str:
        """Return the default name for the instance or class."""
        if instance is None:
            return self.prefix
        self.prefix = instance.__class__.__name__
        self.count_defaults += 1
        return repr(self)

    def __repr__(self) -> str:
        if not self.count_defaults:
            return self.prefix
        return f'{self.prefix}_{self.count_defaults}'


try:
    import pandas as pd


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

except ImportError:
    PandasObject = type(object())
    pd = types.ModuleType('Unreachable module.')


class LiteralStr(str):
    """YAML will attempt to use the pipe style for this class."""


@dataclasses.dataclass(frozen=True)
class Omitted:
    """Class for objects that represent omitted values in an iterable."""
    count: float = math.nan


class StrWithTS(str):
    """A string that adds a timestamp."""
    fmt = '%Y-%m-%dT%H:%M:%S'

    def __new__(cls, suffix: str) -> StrWithTS:

        str_with_timestamp = f'{suffix}.{datetime.datetime.now():{cls.fmt}}'
        return cast(StrWithTS, super().__new__(cls, str_with_timestamp))

    def __str__(self) -> str:
        return self.split('.')[0]


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
    obj_repr = repr(obj) if has_own_repr(obj) else obj.__class__.__name__
    obj_repr = obj_repr.split('(', 1)[0]  # for dataclasses syntax

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
        return {'object': obj_repr} | dict_str

    return obj_repr


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

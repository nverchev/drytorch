""" This module specifies how to represent and dump metadata in a yaml file."""
from typing import Any, Iterable, Optional, Protocol
import datetime
import functools
import types
import yaml  # type: ignore
import numpy as np
import pandas as pd
import torch


class LiteralStr(str):
    pass


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


def represent_datetime(dumper: yaml.Dumper,
                       data: datetime.datetime) -> yaml.Node:
    return dumper.represent_scalar('tag:yaml.org,2002:timestamp',
                                   data.isoformat(timespec='seconds'))


def represent_literal_str(dumper: yaml.Dumper,
                          literal_str: LiteralStr) -> yaml.Node:
    scalar = dumper.represent_str(literal_str)
    scalar.style = '|'
    return scalar


def represent_none(dumper: yaml.Dumper, _: object) -> yaml.Node:
    return dumper.represent_scalar('tag:yaml.org,2002:null', '')


DOTS = LiteralStr('...')
"""Dots represent omitted items in containers."""

yaml.add_representer(datetime, represent_datetime)
yaml.add_representer(LiteralStr, represent_literal_str)
yaml.add_representer(type(None), represent_none)


def has_own_repr(obj: Any) -> bool:
    return not obj.__repr__().endswith(str(hex(id(obj))) + '>')


def limit_size(container: Iterable, max_size: int) -> list:
    # prevents infinite iterators
    if hasattr(container, '__len__'):
        listed = list(container)
        if len(listed) > max_size:
            listed = listed[:max_size // 2] + [DOTS] + listed[-max_size // 2:]
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
            listed.append([DOTS])
    return listed


@functools.singledispatch
def recursive_repr(obj: object, *, max_size: int = 10) -> Any:
    """
    Function that attempts a full documentation of a given object.

    It recursively represents each attribute of the object or the contained
    items in tuple, list, sets and dictionaries,

     limiting their size
     by omitting extra items or,
    in case of arrays from a library, using the library representation.
    .

    Args:
        obj: the object to represent.
        max_size: max length of iterators and arrays.

    Returns:
        a readable representation of the object.
    """
    obj_repr = repr(obj) if has_own_repr(obj) else obj.__class__.__name__

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
def _(obj: Optional[int | float | complex | str], *, max_size: int = 10):
    _not_used = max_size
    return obj


@recursive_repr.register
def _(obj: tuple | list | set, *, max_size: int = 10):
    return type(obj)(recursive_repr(item, max_size=max_size)
                     for item in limit_size(obj, max_size=max_size))


@recursive_repr.register
def _(obj: dict, *, max_size: int = 10):
    return {str(k): recursive_repr(obj.get(k), max_size=max_size)
            for k in limit_size(obj, max_size=max_size)}


@recursive_repr.register
def _(obj: torch.Tensor, *, max_size: int = 10):
    _not_used = max_size
    return recursive_repr(obj.detach().cpu().numpy())


@recursive_repr.register
def _(obj: pd.DataFrame | pd.Series | pd.Index, *, max_size: int = 10):
    with PandasPrintOptions(precision=3,
                            max_rows=max_size,
                            max_columns=max_size):
        return LiteralStr(obj)


@recursive_repr.register
def _(obj: np.ndarray, *, max_size: int = 10):
    with np.printoptions(precision=3,
                         suppress=True,
                         threshold=max_size,
                         edgeitems=max_size // 2):
        return LiteralStr(obj)


@recursive_repr.register(type)
def _(obj, *, max_size: int = 10):
    _not_used = max_size
    return obj.__name__


@recursive_repr.register(types.FunctionType)
def _(obj, *, max_size: int = 10):
    _not_used = max_size
    return obj.__name__

""" This module specifies how to represent and dump metadata in a yaml file."""
import math
from typing import Any, Iterable, Sequence
import functools
import types
import dataclasses

import yaml  # type: ignore
import numpy as np
import numbers
import pandas as pd
import torch

MAX_LENGTH_PLAIN_REPR = 10
"""Sequences longer than this will be represented in flow style by yaml."""
MAX_LENGTH_SHORT_REPR = 10
"""Sequences with strings longer than this will be represented in flow style"""


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


class LiteralStr(str):
    pass


@dataclasses.dataclass(frozen=True)
class Omitted:
    count: float = math.nan


def short_repr(obj: object, max_length: int = MAX_LENGTH_SHORT_REPR) -> bool:
    if not isinstance(obj, str):
        return True
    if isinstance(obj, LiteralStr):
        return False
    return len(obj) <= max_length


def represent_literal_str(dumper: yaml.Dumper,
                          literal_str: LiteralStr) -> yaml.Node:
    return dumper.represent_scalar('tag:yaml.org,2002:str',
                                   literal_str,
                                   style='|')


def represent_sequence(
        dumper: yaml.Dumper,
        sequence: Sequence,
        max_length_for_plain: int = MAX_LENGTH_PLAIN_REPR,
) -> yaml.Node:
    flow_style = False
    len_seq = len(sequence)
    if len_seq <= max_length_for_plain:
        if all(short_repr(elem) for elem in sequence):
            flow_style = True
    return dumper.represent_sequence(tag=u'tag:yaml.org,2002:seq',
                                     sequence=sequence,
                                     flow_style=flow_style)


def represent_omitted(dumper: yaml.Dumper, data: Omitted) -> yaml.Node:
    return dumper.represent_mapping(u'!Omitted',
                                    {'omitted_elements': data.count})


yaml.add_representer(LiteralStr, represent_literal_str)
yaml.add_representer(list, represent_sequence)
yaml.add_representer(tuple, represent_sequence)
yaml.add_representer(set, represent_sequence)
yaml.add_representer(Omitted, represent_omitted)


def has_own_repr(obj: Any) -> bool:
    return not obj.__repr__().endswith(str(hex(id(obj))) + '>')


def limit_size(container: Iterable, max_size: int) -> list:
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
def _(obj: pd.DataFrame | pd.Series | pd.Index,
      *,
      max_size: int = 10) -> LiteralStr:
    with PandasPrintOptions(precision=3,
                            max_rows=max_size,
                            max_columns=max_size):
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

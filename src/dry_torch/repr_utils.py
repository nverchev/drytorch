from typing import Any, Iterable, Optional
import datetime
import functools
import types
import yaml  # type: ignore
import numpy as np
import pandas as pd
import torch


class LiteralStr(str):
    pass


DOTS = LiteralStr('...')


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


yaml.add_representer(datetime, represent_datetime)
yaml.add_representer(LiteralStr, represent_literal_str)
yaml.add_representer(type(None), represent_none)


class PandasPrintOptions:
    """
    Context manager to temporarily set Pandas display options.

    Args:
        precision (int): Number of digits of precision for floating point
        output.
        max_rows (int): Maximum number of rows to display.
        max_columns (int): Maximum number of columns to display.
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
def struc_repr(struc: Any, *, max_size: int = 10) -> Any:
    """
    It attempts full documentation of a complex object.
    It recursively represents each attribute or element of the object.
    To prevent excessive data from being included in the representation,
     it limits the size of containers.
    The readable representation contains only strings and numbers

    Args:
        struc: a complex object with that may contain other objects.
        max_size: limits the size of iterators and arrays.

    Returns:
        a readable representation of the object.
    """
    if isinstance(struc, torch.Tensor):
        struc = struc.detach().cpu()

    if hasattr(struc, 'numpy'):
        try:
            return struc_repr(struc.numpy(), max_size=max_size)
        except TypeError as te:
            msg = 'numpy should be a method without additional parameters'
            raise AttributeError(msg) from te

    dict_attr = getattr(struc, '__dict__', {})
    dict_attr |= {name: getattr(struc, name)
                  for name in getattr(struc, '__slots__', [])}

    dict_str: dict[str, Any] = {}
    for k, v in dict_attr.items():
        if k[0] == '_' or v is struc or v is None:
            continue
        if hasattr(v, '__len__'):
            if not v.__len__():
                continue
        dict_str[k] = struc_repr(v, max_size=max_size)

    if dict_str:
        if has_own_repr(struc):
            return {'repr': struc.__repr__()} | dict_str
        else:
            return {'class': type(struc).__name__} | dict_str

    if hasattr(struc, '__iter__'):
        limit_struc = limit_size(struc, max_size=max_size)
        if hasattr(struc, 'get'):  # assume that the iterable contains keys
            return {str(k): struc_repr(struc.get(k), max_size=max_size)
                    for k in limit_struc}
        struc_list = [struc_repr(item, max_size=max_size)
                      for item in limit_struc]
        if isinstance(struc, (list, set, tuple)):
            return type(struc)(struc_list)
        return struc_list

    return repr(struc) if has_own_repr(struc) else struc.__class__.__name__


@struc_repr.register
def _(struc: Optional[int | float | complex | str], *, max_size: int = 10):
    _not_used = max_size
    return struc


@struc_repr.register
def _(struc: pd.DataFrame | pd.Series | pd.Index, *, max_size: int = 10):
    with PandasPrintOptions(precision=3,
                            max_rows=max_size,
                            max_columns=max_size):
        return LiteralStr(struc)


@struc_repr.register
def _(struc: np.ndarray, *, max_size: int = 10):
    with np.printoptions(precision=3,
                         suppress=True,
                         threshold=max_size,
                         edgeitems=max_size // 2):
        return LiteralStr(struc)


@struc_repr.register(type)
def _(struc, *, max_size: int = 10):
    _not_used = max_size
    return struc.__name__


@struc_repr.register(types.FunctionType)
def _(struc, *, max_size: int = 10):
    _not_used = max_size
    return struc.__name__

from typing import Any, Iterable

import yaml  # type: ignore
import numpy as np
import pandas as pd
import torch


def represent_literal_str(dumper, data):
    scalar = yaml.representer.SafeRepresenter.represent_str(dumper, data)
    scalar.style = '|'
    return scalar


def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')


class LiteralStr(str):
    pass


yaml.add_representer(type(None), represent_none)
yaml.add_representer(LiteralStr, represent_literal_str)


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
            listed = listed[:max_size // 2] + ['...'] + listed[-max_size // 2:]
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
            listed.append(['...'])
    return listed


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
    if struc is None:
        return None

    if isinstance(struc, (int, float, complex, str)):
        return struc

    if isinstance(struc, (type, type(lambda: ...))):
        # no builtin variable for the function class!
        return struc.__name__

    if isinstance(struc, (pd.DataFrame, pd.Series, pd.Index)):
        with PandasPrintOptions(precision=3,
                                max_rows=max_size,
                                max_columns=max_size):
            return LiteralStr(struc)

    if isinstance(struc, np.ndarray):
        with np.printoptions(precision=3,
                             suppress=True,
                             threshold=max_size,
                             edgeitems=max_size // 2):
            return LiteralStr(struc)

    if isinstance(struc, torch.Tensor):
        struc = struc.detach().cpu()

    if hasattr(struc, 'numpy'):
        try:
            return struc_repr(struc.numpy(), max_size=max_size)
        except TypeError as te:
            msg = 'numpy should be a method without additional parameters'
            raise AttributeError(msg) from te

    if hasattr(struc, '__iter__'):
        limit_struc = limit_size(struc, max_size=max_size)
        if hasattr(struc, 'get'):  # assume that the iterable contains keys
            return {str(k): struc_repr(struc.get(k), max_size=max_size)
                    for k in limit_struc}
        struc_list = [struc_repr(item, max_size=max_size)
                      for item in limit_struc]
        if isinstance(struc, (list, set, tuple)):
            return LiteralStr(type(struc)(struc_list))
        return LiteralStr(struc_list)

    dict_attr = getattr(struc, '__dict__', {})
    dict_attr |= {name: getattr(struc, name)
                  for name in getattr(struc, '__slots__', [])}
    dict_attr = {k: struc_repr(v, max_size=max_size)
                 for k, v in dict_attr.items()
                 if k[0] != '_' and v is not struc}
    if dict_attr:
        if has_own_repr(struc):
            return {'repr': struc.__repr__()} | dict_attr
        else:
            return {'class': type(struc).__name__} | dict_attr

    return repr(struc) if has_own_repr(struc) else struc.__class__.__name__

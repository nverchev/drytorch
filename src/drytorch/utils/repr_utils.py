"""Module containing utilities to extract readable representations."""

from __future__ import annotations

import dataclasses
import datetime
import functools
import itertools
import math
import numbers
import types

from collections.abc import Hashable, Iterable
from itertools import count
from typing import Any

import numpy as np
import torch


class CreatedAtMixin:
    """Mixin saving instantiation timestamp."""

    ts_fmt = '%Y-%m-%d@%Hh%Mm%Ss'

    def __init__(self, *args, **kwargs) -> None:
        """Constructor."""
        self._created_at = datetime.datetime.now()
        super().__init__(*args, **kwargs)

    @property
    def created_at(self) -> datetime.datetime:
        """Read-only timestamp."""
        return self._created_at

    @property
    def created_at_str(self) -> str:
        """Read-only timestamp."""
        return self._created_at.strftime(self.ts_fmt)


class DefaultName:
    """Add a counter to a prefix."""

    def __init__(self) -> None:
        """Constructor."""
        self._prefixes: dict[str, count[int]] = {}

    def __get__(self, instance: Any, objtype: type | None = None) -> str:
        """Return the default name for the instance or class."""
        return instance.__name  # pylint: disable=protected-access

    def __set__(self, instance: Any, value: str) -> None:
        """Return the default name for the instance or class."""
        value = value if value else instance.__class__.__name__
        count_iter = self._prefixes.setdefault(value, itertools.count())
        if count_value := next(count_iter):
            value = f'{value}_{count_value}'
        # pylint: disable=unused-private-member, protected-access
        instance.__name = value
        return


try:
    import pandas as pd

except (ImportError, ModuleNotFoundError):
    PandasObject = type(object())
    pd = types.ModuleType('Unreachable module.')

else:

    class PandasPrintOptions:
        """Context manager to temporarily set Pandas display options.

        Args:
            precision: number of digits of precision for floating point output.
            max_rows: maximum number of rows to display.
            max_columns: maximum number of columns to display.
        """

        def __init__(
                self, precision: int = 3, max_rows: int = 10,
                max_columns: int = 10
        ) -> None:
            """Constructor.

            Args:
                precision: see Pandas docs.
                max_rows: see Pandas docs.
                max_columns: see Pandas docs.
            """
            self._options: dict[str, int] = {
                'display.precision': precision,
                'display.max_rows': max_rows,
                'display.max_columns': max_columns,
            }
            self._original_options: dict[str, Any] = {}

        def __enter__(self) -> None:
            """Temporarily modify settings."""
            self._original_options.update(
                {key: pd.get_option(key) for key in self._options}
            )
            for key, value in self._options.items():
                pd.set_option(key, value)

        def __exit__(
                self,
                exc_type: None = None,
                exc_val: None = None,
                exc_tb: None = None,
        ) -> None:
            """Restore original settings."""
            for key, value in self._original_options.items():
                pd.set_option(key, value)


    PandasObject = pd.core.base.PandasObject  # type: ignore


class LiteralStr(str):
    """YAML will attempt to use the pipe style for this class."""


@dataclasses.dataclass(frozen=True)
class Omitted:
    """Represent omitted values in an iterable object.

    Attributes:
        count: how many elements have been omitted. Defaults to NAN (unknown).
    """

    count: float = math.nan


def has_own_repr(obj: Any) -> bool:
    """Indicate whether __repr__ has been overridden."""
    return not repr(obj).endswith(str(hex(id(obj))) + '>')


def limit_size(container: Iterable[Any], max_size: int) -> list[Any]:
    """Limit the size of iterables and adds an Omitted object."""
    # prevents infinite iterators
    if hasattr(container, '__len__'):
        listed = list(container)
        if len(listed) > max_size:
            omitted = [Omitted(len(listed) - max_size)]
            listed = (
                    listed[: max_size // 2] + omitted + listed[-max_size // 2:]
            )

    else:
        listed = []
        iter_container = iter(container)
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
    """Function that attempts a hierarchical representation of a given object.

    It recursively represents each attribute of the object or the contained
    items in tuples, lists, sets, and dictionaries. The latter structures are
    limited in size by keeping max_size elements and replacing the others with
    an Omitted instance.

    Arrays are represented using pandas and numpy array representation. Numbers
    are returned as they are or converted to python types.

    Args:
        obj: the object to represent.
        max_size: max length of iterators and arrays.

    Returns:
        A readable representation of the object.
    """
    class_name = obj.__class__.__name__
    dict_attr = getattr(obj, '__dict__', {})
    if not dict_attr:
        dict_attr = {
            name: getattr(obj, name) for name in getattr(obj, '__slots__', [])
        }

    dict_str: dict[str, Any] = {}
    for k, v in dict_attr.items():
        if k[0] == '_' or v is obj or v is None:
            continue

        if hasattr(v, '__len__'):
            try:
                len_v = len(v)
            except (TypeError, NotImplementedError):
                continue
            else:
                if not len_v:
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
    if item_method := getattr(obj, 'item', None):
        try:
            obj = item_method()
        except (TypeError, NotImplementedError):
            pass
    _not_used = max_size
    return obj


@recursive_repr.register
def _(obj: tuple, *, max_size: int = 10) -> tuple[Any, ...]:
    return tuple(
        recursive_repr(item, max_size=max_size)
        for item in limit_size(obj, max_size=max_size)
    )


@recursive_repr.register
def _(obj: list, *, max_size: int = 10) -> list[Any]:
    return [
        recursive_repr(item, max_size=max_size)
        for item in limit_size(obj, max_size=max_size)
    ]


@recursive_repr.register
def _(obj: set, *, max_size: int = 10) -> set[Hashable]:
    return {
        recursive_repr(item, max_size=max_size)
        for item in limit_size(obj, max_size=max_size)
    }


@recursive_repr.register
def _(obj: dict, *, max_size: int = 10) -> dict[str, Any]:
    out_dict: dict[str, Any] = {
        str(key): recursive_repr(value, max_size=max_size)
        for key, value in list(obj.items())[:max_size]
    }
    if len(obj) > max_size:
        out_dict['...'] = Omitted(len(obj) - max_size)
    return out_dict


@recursive_repr.register
def _(obj: torch.Tensor, *, max_size: int = 10) -> LiteralStr:
    _not_used = max_size
    return recursive_repr(obj.detach().cpu().numpy(), max_size=max_size)


@recursive_repr.register
def _(
        obj: PandasObject,  # type: ignore
        *,
        max_size: int = 10,
) -> LiteralStr:
    # only called when Pandas is imported
    with PandasPrintOptions(max_rows=max_size, max_columns=max_size):
        return LiteralStr(obj)


@recursive_repr.register
def _(obj: np.ndarray, *, max_size: int = 10) -> LiteralStr:
    size_factor = 2 ** (+obj.ndim - 1)
    with np.printoptions(
            precision=3,
            suppress=True,
            threshold=max_size // size_factor,
            edgeitems=max_size // (size_factor * 2),
    ):
        return LiteralStr(obj)


@recursive_repr.register(type)
def _(obj, *, max_size: int = 10) -> str:
    _not_used = max_size
    return obj.__name__


@recursive_repr.register(types.FunctionType)
def _(obj, *, max_size: int = 10) -> str:
    _not_used = max_size
    return obj.__name__

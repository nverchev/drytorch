"""Module containing utilities to extract readable representations.

Attributes:
    MAX_DEPTH: Max number of recursions when representing an object.
    MAX_REPR_SIZE: Max representation size for iterators and arrays.
    INCLUDE_PROPERTIES: Whether to evaluate properties and represent them.
"""

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
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeAlias

import numpy as np
import torch


if TYPE_CHECKING:
    import numpy.typing as npt

    from pandas.core.generic import NDFrame

    GenericDict: TypeAlias = dict[Hashable, Any]
    GenericList: TypeAlias = list[Any]
    GenericSet: TypeAlias = set[Any]
    GenericTuple: TypeAlias = tuple[Any, ...]
    ndarray: TypeAlias = npt.NDArray[Any]

else:
    from numpy.core.records import ndarray

    GenericList = list
    GenericDict = dict
    GenericSet = set
    GenericTuple = tuple

MAX_DEPTH: int = 10
MAX_REPR_SIZE: int = 10
INCLUDE_PROPERTIES: bool = False


class CreatedAtMixin:
    """Mixin saving instantiation timestamp."""

    ts_fmt: ClassVar = '%Y-%m-%d@%Hh%Mm%Ss'

    def __init__(self, *args, **kwargs) -> None:
        """Constructor."""
        self._created_at: Final = datetime.datetime.now()
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
        return instance.__name

    def __set__(self, instance: Any, value: str) -> None:
        """Return the default name for the instance or class."""
        value = value if value else instance.__class__.__name__
        count_iter = self._prefixes.setdefault(value, itertools.count())
        if count_value := next(count_iter):
            value = f'{value}_{count_value}'
        instance.__name = value
        return


class LiteralStr(str):
    """YAML will attempt to use the pipe style for this class."""


@dataclasses.dataclass(frozen=True)
class Omitted:
    """Represent omitted values in an iterable object.

    Attributes:
        count: how many elements have been omitted. Defaults to NAN (unknown).
    """

    count: float = math.nan


@functools.singledispatch
def recursive_repr(obj: object, *, depth: int = MAX_DEPTH) -> Any:
    """Create a hierarchical representation of an object.

    It recursively represents each attribute of the object or the contained
    items in tuples, lists, sets, and dictionaries. The latter structures are
    limited in size by limiting the number of elements and replacing the others
    with an Omitted instance. Arrays are represented using native representation
    Numbers are returned as they are or converted to built-in types.

    Args:
        obj: The object to represent
        depth: Maximum recursion depth allowed

    Returns:
        A readable representation of the object
    """
    class_name = obj.__class__.__name__
    if depth == 0:
        return repr(obj) if _has_own_repr(obj) else class_name

    attributes = _get_object_attributes(obj)
    result_attrs = {}
    for key, value in attributes.items():
        if _should_skip_attribute(key, value, obj):
            continue

        result_attrs[key] = recursive_repr(value, depth=depth - 1)

    if result_attrs:
        return {'class': class_name, **dict(sorted(result_attrs.items()))}

    return repr(obj) if _has_own_repr(obj) else class_name


@recursive_repr.register
def _(obj: Omitted, *, depth: int = 10) -> Omitted:
    _not_used = depth
    return obj


@recursive_repr.register
def _(obj: str, *, depth: int = 10) -> str:
    _not_used = depth
    return obj


@recursive_repr.register
def _(obj: None, *, depth: int = 10) -> None:
    _not_used = depth
    return obj


@recursive_repr.register
def _(obj: numbers.Number, *, depth: int = 10) -> numbers.Number:
    if item_method := getattr(obj, 'item', None):
        try:
            obj = item_method()
        except (TypeError, NotImplementedError):
            pass
    _not_used = depth
    return obj


@recursive_repr.register
def _(obj: GenericTuple, *, depth: int = 1) -> tuple[Any, ...]:
    return tuple(
        recursive_repr(item, depth=depth - 1) for item in _limit_size(obj)
    )


@recursive_repr.register
def _(obj: GenericList, *, depth: int = 10) -> list[Any]:
    return [recursive_repr(item, depth=depth - 1) for item in _limit_size(obj)]


@recursive_repr.register
def _(obj: GenericSet, *, depth: int = 10) -> set[Hashable]:
    return {recursive_repr(item, depth=depth - 1) for item in _limit_size(obj)}


@recursive_repr.register
def _(obj: GenericDict, *, depth: int = 10) -> dict[str, Any]:
    out_dict: dict[str, Any] = {
        str(key): recursive_repr(value, depth=depth - 1)
        for key, value in list(obj.items())[:MAX_REPR_SIZE]
    }
    if len(obj) > MAX_REPR_SIZE:
        out_dict['...'] = Omitted(len(obj) - MAX_REPR_SIZE)
    return out_dict


@recursive_repr.register
def _(obj: torch.Tensor, *, depth: int = 10) -> LiteralStr:
    _not_used = depth
    return recursive_repr(obj.detach().cpu().numpy())


@recursive_repr.register
def _(obj: ndarray, *, depth: int = 10) -> LiteralStr:
    size_factor = 2 ** (+obj.ndim - 1)
    with np.printoptions(
        precision=3,
        suppress=True,
        threshold=MAX_REPR_SIZE // size_factor,
        edgeitems=MAX_REPR_SIZE // (size_factor * 2),
    ):
        _not_used = depth
        return LiteralStr(obj)


@recursive_repr.register(type)
def _(obj, *, depth: int = 10) -> str:
    _not_used = depth
    return obj.__name__


@recursive_repr.register(types.FunctionType)
def _(obj, *, depth: int = 10) -> str:
    _not_used: int = depth
    return obj.__name__


def _get_object_attributes(obj: object) -> dict[str, Any]:
    """Extract all relevant attributes from an object."""
    # Get instance attributes
    attributes = getattr(obj, '__dict__', {}).copy()

    # Add slot attributes if no __dict__
    if not attributes:
        slot_names = getattr(obj, '__slots__', [])
        attributes = {name: getattr(obj, name, None) for name in slot_names}

    # Add properties if enabled
    if INCLUDE_PROPERTIES:
        attributes.update(_get_object_properties(obj))

    return attributes


def _get_object_properties(obj: object) -> dict[str, Any]:
    """Extract property values from an object."""
    properties = {}

    for cls in reversed(obj.__class__.__mro__):
        for name, attr in cls.__dict__.items():
            if isinstance(attr, property):
                try:
                    properties[name] = getattr(obj, name)
                except Exception as e:
                    properties[name] = str(e)

    return properties


def _has_own_repr(obj: Any) -> bool:
    """Indicate whether __repr__ has been overridden."""
    repr_str = repr(obj).lower()  # Windows use the upper case
    hex_id = hex(id(obj))[2:]  # remove '0x' prefix
    return not repr_str.endswith(hex_id + '>')


def _limit_size(container: Iterable[Any]) -> list[Any]:
    """Limit the size of iterables and adds an Omitted object."""
    # prevents infinite iterators
    if hasattr(container, '__len__'):
        listed = list(container)
        if len(listed) > MAX_REPR_SIZE:
            omitted = [Omitted(len(listed) - MAX_REPR_SIZE)]
            listed = (
                listed[: MAX_REPR_SIZE // 2]
                + omitted
                + listed[-MAX_REPR_SIZE // 2 :]
            )

    else:
        listed = []
        iter_container = iter(container)
        for _ in range(MAX_REPR_SIZE):
            try:
                value = next(iter_container)
                listed.append(value)
            except StopIteration:
                break

        else:
            listed.append([Omitted()])

    return listed


def _should_skip_attribute(key: str, value: Any, parent_obj: object) -> bool:
    """Determine if an attribute should be skipped during representation."""
    if key.startswith('_'):
        return True

    if value is parent_obj or value is None:
        return True

    if hasattr(value, '__len__'):
        try:
            if len(value) == 0:
                return True
        except (TypeError, NotImplementedError):
            return True

    return False


try:
    import pandas as pd

except (ImportError, ModuleNotFoundError):
    pass

else:

    class PandasPrintOptions:
        """Context manager to temporarily set Pandas display options.

        Args:
            precision: number of digits of precision for floating point output.
            max_rows: maximum number of rows to display.
            max_columns: maximum number of columns to display.
        """

        def __init__(
            self, precision: int = 3, max_rows: int = 10, max_columns: int = 10
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

    from pandas.core.generic import NDFrame

    @recursive_repr.register
    def _(obj: NDFrame, *, depth: int = 10) -> LiteralStr:
        # only called when Pandas is imported
        _not_used = depth
        with PandasPrintOptions(
            max_rows=MAX_REPR_SIZE, max_columns=MAX_REPR_SIZE
        ):
            return LiteralStr(obj)

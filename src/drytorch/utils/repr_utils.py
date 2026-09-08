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

from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeAlias, overload

import numpy as np
import torch

from torch import _tensor_str
from typing_extensions import override


__all__ = [
    'INCLUDE_PROPERTIES',
    'MAX_DEPTH',
    'MAX_REPR_SIZE',
    'recursive_repr',
]

if TYPE_CHECKING:
    import numpy.typing as npt

    from pandas.core.generic import NDFrame

    GenericDict: TypeAlias = dict[Hashable, Any]
    GenericList: TypeAlias = list[Any]
    GenericSet: TypeAlias = set[Any]
    GenericTuple: TypeAlias = tuple[Any, ...]
    ndarray: TypeAlias = npt.NDArray[Any]

else:
    from numpy import ndarray

    GenericList = list
    GenericDict = dict
    GenericSet = set
    GenericTuple = tuple

MAX_DEPTH: int = 10
MAX_REPR_SIZE: int = 10
INCLUDE_PROPERTIES: bool = False


class CreatedAtMixin:
    """Mixin saving instantiation timestamp.

    Attributes:
        ts_fmt: timestamp format string.
    """

    ts_fmt: ClassVar[str] = '%Y-%m-%d@%Hh%Mm%Ss'

    _created_at: datetime.datetime

    def __init__(self, *args, **kwargs) -> None:
        """Initialize."""
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
    """Add a counter to a prefix.

    Attributes:
        _prefixes: dictionary mapping prefixes to their counters.
    """

    _prefixes: dict[str, itertools.count[int]]

    def __init__(self) -> None:
        """Initialize."""
        self._prefixes = {}

    @overload
    def __get__(self, instance: None, objtype: type | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Any, objtype: type | None = None) -> str: ...

    def __get__(self, instance: Any, objtype: type | None = None) -> Self | str:
        """Return descriptor on class access or default name on instance."""
        if instance is None:
            return self
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

    @override
    def __add__(self, other: str | LiteralStr) -> LiteralStr:
        out = super().__add__(other)
        return LiteralStr(out)


@dataclasses.dataclass(frozen=True)
class Omitted:
    """Represent omitted values in an iterable object.

    Attributes:
        count: how many elements have been omitted. Defaults to NAN (unknown).
    """

    count: float = math.nan


class TorchPrintOptions:
    """Context manager to temporarily set PyTorch tensor print options.

    Args:
        precision: number of digits of precision for floating point output.
        threshold: max number of elements before triggering summarization.
        edgeitems: number of elements per dimension when summarizing.
        sci_mode: whether to use scientific notation.
    """

    _options: dict[str, Any]
    _original_options: dict[str, Any]

    def __init__(
        self,
        precision: int = 3,
        threshold: int = 10,
        edgeitems: int = 3,
        sci_mode: bool = False,
    ) -> None:
        """Initialize.

        Args:
            precision: see torch.set_printoptions docs.
            threshold: see torch.set_printoptions docs.
            edgeitems: see torch.set_printoptions docs.
            sci_mode: see torch.set_printoptions docs.
        """
        self._options = {
            'precision': precision,
            'threshold': threshold,
            'edgeitems': edgeitems,
            'sci_mode': sci_mode,
        }
        self._original_options = {}
        return

    def __enter__(self) -> None:
        """Temporarily modify settings."""
        self._original_options.update(_tensor_str.get_printoptions())
        torch.set_printoptions(**self._options)
        return

    def __exit__(
        self,
        exc_type: None = None,
        exc_val: None = None,
        exc_tb: None = None,
    ) -> None:
        """Restore original settings."""
        torch.set_printoptions(**self._original_options)
        return


@functools.singledispatch
def _dispatch_repr(
    obj: object,
    *,
    depth: int,
    _visited: set[int],
) -> Any:
    class_name = obj.__class__.__name__
    if depth > 0:
        attributes = _get_object_attributes(obj)
        result_attrs = {}
        for key, value in attributes.items():
            if _should_skip_attribute(key, value, obj):
                continue

            result_attrs[key] = recursive_repr(
                value, depth=depth - 1, _visited=_visited
            )

        if result_attrs:
            return {'class': class_name, **dict(sorted(result_attrs.items()))}

    return repr(obj) if _has_own_repr(obj) else class_name


@_dispatch_repr.register
def _(obj: Omitted, *, depth: int = 10, _visited: set[int]) -> Omitted:
    _not_used = depth
    return obj


@_dispatch_repr.register
def _(obj: str, *, depth: int = 10, _visited: set[int]) -> str:
    _not_used = depth
    return obj


@_dispatch_repr.register
def _(obj: None, *, depth: int = 10, _visited: set[int]) -> None:
    _not_used = depth
    return obj


@_dispatch_repr.register
def _(
    obj: numbers.Number, *, depth: int = 10, _visited: set[int]
) -> numbers.Number:
    if item_method := getattr(obj, 'item', None):
        try:
            obj = item_method()
        except (TypeError, NotImplementedError):
            pass

    _not_used = depth
    return obj


@_dispatch_repr.register
def _(
    obj: GenericTuple, *, depth: int = 1, _visited: set[int]
) -> tuple[Any, ...]:
    return tuple(
        recursive_repr(item, depth=depth - 1, _visited=_visited)
        for item in _limit_size(obj)
    )


@_dispatch_repr.register
def _(obj: GenericList, *, depth: int = 10, _visited: set[int]) -> list[Any]:
    return [
        recursive_repr(item, depth=depth - 1, _visited=_visited)
        for item in _limit_size(obj)
    ]


@_dispatch_repr.register
def _(obj: GenericSet, *, depth: int = 10, _visited: set[int]) -> set[Hashable]:
    return {
        recursive_repr(item, depth=depth - 1, _visited=_visited)
        for item in _limit_size(obj)
    }


@_dispatch_repr.register
def _(
    obj: GenericDict, *, depth: int = 10, _visited: set[int]
) -> dict[str, Any]:
    out_dict: dict[str, Any] = {
        str(key): recursive_repr(value, depth=depth - 1, _visited=_visited)
        for key, value in list(obj.items())[:MAX_REPR_SIZE]
    }
    if len(obj) > MAX_REPR_SIZE:
        out_dict['...'] = Omitted(len(obj) - MAX_REPR_SIZE)
    return out_dict


@_dispatch_repr.register
def _(obj: torch.Tensor, *, depth: int = 10, _visited: set[int]) -> LiteralStr:
    _not_used = depth
    _not_used2 = _visited
    size_str = f'Tensor of size {tuple(obj.shape)}, dtype={obj.dtype}\n'
    with TorchPrintOptions(
        threshold=MAX_REPR_SIZE, edgeitems=max(MAX_REPR_SIZE // 4, 1)
    ):
        return LiteralStr(size_str) + LiteralStr(repr(obj))


@_dispatch_repr.register
def _(obj: ndarray, *, depth: int = 10, _visited: set[int]) -> LiteralStr:
    size_factor = 2 ** (+obj.ndim - 1)
    size_str = f'Array of size {obj.shape}\n'
    with np.printoptions(
        precision=3,
        suppress=True,
        threshold=MAX_REPR_SIZE // size_factor,
        edgeitems=max(MAX_REPR_SIZE // (size_factor * 2), 1),
    ):
        _not_used = depth
        return LiteralStr(size_str) + LiteralStr(obj)


@_dispatch_repr.register(type)
def _(obj, *, depth: int = 10, _visited: set[int]) -> str:
    _not_used = depth
    return obj.__name__


@_dispatch_repr.register(types.FunctionType)
def _(obj, *, depth: int = 10, _visited: set[int]) -> str:
    _not_used: int = depth
    return obj.__name__


def recursive_repr(
    obj: object,
    *,
    depth: int | None = None,
    _visited: set[int] | None = None,
) -> Any:
    """Create a hierarchical representation of an object with cycle detection.

    It recursively represents each attribute of the object or the contained
    items in tuples, lists, sets, and dictionaries. The latter structures are
    limited in size by limiting the number of elements and replacing the others
    with an Omitted instance. Arrays are represented using native representation
    Numbers are returned as they are or converted to built-in types.

    Args:
        obj: The object to represent
        depth: Maximum recursion depth allowed (defaults to MAX_DEPTH)
        _visited: Set of object IDs already visited (for cycle detection)

    Returns:
        A readable representation of the object
    """
    if _visited is None:
        _visited = set()

    if depth is None:
        depth = MAX_DEPTH

    if not isinstance(obj, (str, numbers.Number, type, types.FunctionType)):
        obj_id = id(obj)
        if obj_id in _visited:
            return {'duplicated': _dispatch_repr(obj, depth=1, _visited=set())}

        _visited.add(obj_id)

    return _dispatch_repr(obj, depth=depth, _visited=_visited)


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


def _limit_size(container: Collection[Any]) -> list[Any]:
    """Limit the size of a container and add an Omitted object."""
    len_container = len(container)
    if len_container <= MAX_REPR_SIZE:
        return list(container)

    omitted = Omitted(len_container - MAX_REPR_SIZE)
    if isinstance(container, Sequence):
        head = MAX_REPR_SIZE // 2
        tail = MAX_REPR_SIZE - head
        tail_items = container[-tail:] if tail else ()
        return [*container[:head], omitted, *tail_items]

    listed = list(itertools.islice(container, MAX_REPR_SIZE))
    listed.append(omitted)
    return listed


def _should_skip_attribute(key: str, value: Any, parent_obj: object) -> bool:
    """Determine if an attribute should be skipped during representation."""
    if key.startswith('_') or value is None:
        return True

    if hasattr(value, '__len__'):
        try:
            len_value = len(value)
        except (TypeError, NotImplementedError):
            ...
        else:
            return len_value == 0

    return False


try:
    import pandas as pd

except (ImportError, ModuleNotFoundError):
    pass

else:
    from pandas.core.generic import NDFrame

    class PandasPrintOptions:
        """Context manager to temporarily set Pandas display options.

        Args:
            precision: number of digits of precision for floating point output.
            max_rows: maximum number of rows to display.
            max_columns: maximum number of columns to display.
        """

        _options: dict[str, int]
        _original_options: dict[str, Any]

        def __init__(
            self, precision: int = 3, max_rows: int = 10, max_columns: int = 10
        ) -> None:
            """Initialize.

            Args:
                precision: see Pandas docs.
                max_rows: see Pandas docs.
                max_columns: see Pandas docs.
            """
            self._options = {
                'display.precision': precision,
                'display.max_rows': max_rows,
                'display.max_columns': max_columns,
            }
            self._original_options = {}
            return

        def __enter__(self) -> None:
            """Temporarily modify settings."""
            self._original_options.update(
                {key: pd.get_option(key) for key in self._options}
            )
            for key, value in self._options.items():
                pd.set_option(key, value)

            return

        def __exit__(
            self,
            exc_type: None = None,
            exc_val: None = None,
            exc_tb: None = None,
        ) -> None:
            """Restore original settings."""
            for key, value in self._original_options.items():
                pd.set_option(key, value)

            return

    @_dispatch_repr.register
    def _(obj: NDFrame, *, depth: int = 10, _visited: set[int]) -> LiteralStr:
        # only defined when Pandas is imported
        _not_used = depth
        with PandasPrintOptions(
            max_rows=MAX_REPR_SIZE, max_columns=MAX_REPR_SIZE
        ):
            return LiteralStr(obj)

"""Functions for nested containers."""

import copy
import dataclasses

from collections.abc import Callable, MutableMapping, MutableSequence
from typing import Final, TypeVar, overload

import torch

from drytorch.core import exceptions


__all__ = [
    'apply',
    'apply_cpu_detach',
    'apply_to',
]

_T = TypeVar('_T')
_C = TypeVar('_C')

_MISSING = object()
_PASSTHROUGH_TYPES: Final = (str, bytes, type(None))


@overload
def apply(
    obj: _C, /, expected_type: type[_T], func: Callable[[_T], _T]
) -> _C: ...


@overload
def apply(
    obj: _T, /, expected_type: type[_T], func: Callable[[_T], _T]
) -> _T: ...


def apply(
    obj: _C, /, expected_type: type[_T], func: Callable[[_T], _T]
) -> _C | _T:
    """Look for an expected type and apply a given function.

    The implementation is similar to default_convert in
    github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py.
    It transforms objects of the expected type, traverses mappings, lists,
    tuples, named tuples and dataclass instances, and returns strings, bytes
    and None unchanged. Every other unrecognised type raises
    FuncNotApplicableError.

    Args:
        obj: target or object containing other containers and target objects.
        expected_type: the type of the target object to modify.
        func: a function that modifies target objects of the expected type.

    Returns:
        the modified target or container with the modified target objects.

    Raises:
        FuncNotApplicableError: if the object is of an unexpected type.
        NamedTupleOnlyError: if the attempt of copying a tuple failed.
    """
    if isinstance(obj, expected_type):
        return func(obj)

    if isinstance(obj, _PASSTHROUGH_TYPES):
        return obj

    if isinstance(obj, MutableMapping):
        mapping = copy.copy(obj)
        mapping.update(
            {key: apply(item, expected_type, func) for key, item in obj.items()}
        )
        return mapping

    if isinstance(obj, MutableSequence):
        sequence = copy.copy(obj)
        for i, value in enumerate(obj):
            sequence[i] = apply(value, expected_type, func)

        return sequence

    if isinstance(obj, tuple):
        new = (apply(item, expected_type, func) for item in obj)
        if obj.__class__ is tuple:
            return obj.__class__(new)

        try:
            return obj.__class__(*new)
        except TypeError as te:
            raise exceptions.NamedTupleOnlyError(obj.__class__.__name__) from te

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _apply_dataclass(obj, expected_type, func)

    # a callable that is not a function has no __name__
    func_name = getattr(func, '__name__', type(func).__name__)
    raise exceptions.FuncNotApplicableError(func_name, type(obj).__name__)


def apply_to(obj: _C, /, device: torch.device) -> _C:
    """Change the device of tensors inside a container.

    Args:
        obj: object containing other containers and tensors.
        device: the target device.

    Returns:
        the container with the target tensors on the target device.
    """
    non_blocking = device != torch.device('cpu')

    def _to_device(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=non_blocking)

    _to_device.__name__ = 'apply_to'
    return apply(obj, torch.Tensor, _to_device)


def apply_cpu_detach(obj: _C, /) -> _C:
    """Detach and store in cpu the tensors inside a container.

    Args:
        obj: object containing other containers and tensors.

    Returns:
        the container with the target tensors on cpu.
    """

    def _cpu_detach(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu()

    _cpu_detach.__name__ = 'apply_cpu_detach'
    return apply(obj, torch.Tensor, _cpu_detach)


def _apply_dataclass(
    obj: _C, expected_type: type[_T], func: Callable[[_T], _T]
) -> _C:
    names: list[str] = []
    names.extend(getattr(obj, '__dict__', {}))
    for cls in type(obj).__mro__:
        names.extend(s for s in getattr(cls, '__slots__', ()) if s not in names)

    new_obj = copy.copy(obj)
    for name in names:
        value = getattr(obj, name, _MISSING)
        if value is _MISSING:
            continue

        object.__setattr__(new_obj, name, apply(value, expected_type, func))

    return new_obj

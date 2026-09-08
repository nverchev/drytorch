"""Module containing functions for nested containers."""

import copy

from collections.abc import Callable, MutableMapping, MutableSequence
from typing import TypeVar, overload

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


@overload
def recursive_apply(
    obj: _C, expected_type: type[_T], func: Callable[[_T], _T]
) -> _C: ...


@overload
def recursive_apply(
    obj: _T, expected_type: type[_T], func: Callable[[_T], _T]
) -> _T: ...


def recursive_apply(
    obj: _C, expected_type: type[_T], func: Callable[[_T], _T]
) -> _C | _T:
    """Look for an expected type and apply a given function.

    The implementation is similar to default_convert in
    github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py.
    It makes a deepcopy of a MutableMapping or MutableSequence container and
    modifies the elements of the expected type using the functions or act
    recursively on other containers. If obj is a namedtuple, the
    function uses the class constructor to create a new instance with the
    modified elements. Note that when applied after default_convert, the only
    objects of type tuple are namedtuple classes.

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

    if isinstance(obj, MutableMapping):
        mapping = copy.copy(obj)
        mapping.update(
            {
                key: recursive_apply(item, expected_type, func)
                for key, item in obj.items()
            }
        )
        return mapping  # type: ignore

    if isinstance(obj, MutableSequence):
        sequence = copy.copy(obj)
        for i, value in enumerate(obj):
            sequence[i] = recursive_apply(value, expected_type, func)

        return sequence  # type: ignore

    if isinstance(obj, tuple):
        new = (recursive_apply(item, expected_type, func) for item in obj)
        if obj.__class__ is tuple:
            return obj.__class__(new)  # type: ignore

        try:
            return obj.__class__(*new)  # type: ignore
        except TypeError as te:
            raise exceptions.NamedTupleOnlyError(obj.__class__.__name__) from te

    raise exceptions.FuncNotApplicableError(
        func.__name__, obj.__class__.__name__
    )


def apply(obj: _C, expected_type: type[_T], func: Callable[[_T], _T]) -> _C:
    """Extend recursive_apply supports.

    If the input has attributes, it calls recursive_apply, creates a new
    instance and sets the attributes of a new instance to the new values.

    Args:
        obj: object containing other containers and target objects.
        expected_type: the type of the target object to modify.
        func: a function that modifies target objects of the expected type.

    Returns:
        the container with the modified target objects.
    """
    if isinstance(obj, type):
        return recursive_apply(obj, expected_type=expected_type, func=func)

    names: list[str] = []
    names.extend(getattr(obj, '__dict__', {}))
    for cls in type(obj).__mro__:
        names.extend(s for s in getattr(cls, '__slots__', ()) if s not in names)

    if not names:
        return recursive_apply(obj, expected_type=expected_type, func=func)

    new_obj = copy.copy(obj)
    for name in names:
        value = getattr(obj, name, _MISSING)
        if value is _MISSING:
            continue

        object.__setattr__(
            new_obj, name, recursive_apply(value, expected_type, func)
        )

    return new_obj


def apply_to(obj: _C, device: torch.device) -> _C:
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

    return apply(obj, expected_type=torch.Tensor, func=_to_device)


def apply_cpu_detach(obj: _C) -> _C:
    """Detach and store in cpu the tensors inside a container.

    Args:
        obj: object containing other containers and tensors.

    Returns:
        the container with the target tensors on cpu.
    """

    def _cpu_detach(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu()

    return apply(obj, expected_type=torch.Tensor, func=_cpu_detach)

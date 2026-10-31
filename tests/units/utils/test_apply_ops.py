"""Tests for the ``apply_ops`` module."""

import dataclasses

from typing import NamedTuple

import torch

import pytest

from drytorch.core import exceptions
from drytorch.utils.apply_ops import (
    _PASSTHROUGH_TYPES,
    apply,
    apply_cpu_detach,
    apply_to,
)


class _TorchTuple(NamedTuple):
    """NamedTuple for testing apply."""

    one: torch.Tensor
    two: torch.Tensor


class _TorchLikeTuple(NamedTuple):
    """NamedTuple subclass for testing recursive_to."""

    tensor: torch.Tensor
    tensor_lst: list[torch.Tensor]


class _NamedTupleWithString(NamedTuple):
    """NamedTuple containing a string."""

    text: str


class _BaseTestClass:
    """Base test class."""

    t1: torch.Tensor
    t2: torch.Tensor


@dataclasses.dataclass()
class DataClass(_BaseTestClass):
    """Test dataclass."""

    t1: torch.Tensor
    t2: torch.Tensor


@dataclasses.dataclass(slots=True)
class SlottedDataClass(_BaseTestClass):
    """Test dataclass with slots."""

    t1: torch.Tensor
    t2: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FrozenDataClass(_BaseTestClass):
    """Test dataclass with slots."""

    t1: torch.Tensor
    t2: torch.Tensor


@dataclasses.dataclass
class InitFalseDataClass(_BaseTestClass):
    """Test dataclass with an init method."""

    t1: torch.Tensor
    t2: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        """Construct a second field."""
        self.t2 = self.t1 + 1
        return


@dataclasses.dataclass
class InitVarDataClass(_BaseTestClass):
    """Test dataclass with an InitVar."""

    t1: torch.Tensor
    t2: torch.Tensor = dataclasses.field(init=False)
    offset: dataclasses.InitVar[int] = 1

    def __post_init__(self, offset: int) -> None:
        """Construct a second field."""
        self.t2 = self.t1 + offset
        return


@dataclasses.dataclass
class DataClassWithInit(_BaseTestClass):
    """Test dataclass with an init=False field."""

    def __init__(self, t1: torch.Tensor, t2: torch.Tensor):
        """Initialize."""
        self.t1 = t1
        self.t2 = t2
        return


class _BaseWithSlot:
    """Base class with a slot."""

    __slots__ = ('extra',)

    extra: torch.Tensor


@dataclasses.dataclass
class _DerivedWithBaseSlot(_BaseWithSlot):
    """Dataclass inheriting from a base with a slot."""

    t1: torch.Tensor


@dataclasses.dataclass
class _DataClassWithManualSlot:
    """Dataclass with a manual __slots__ entry that is not a field."""

    __slots__ = ('a', 'cache')

    a: torch.Tensor


@dataclasses.dataclass
class _Inner:
    """Dataclass nested in another one."""

    t1: torch.Tensor


@dataclasses.dataclass(slots=True)
class _SlottedInner:
    """Dataclass with slots nested in another one."""

    t1: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _FrozenInner:
    """Frozen dataclass nested in another one."""

    t1: torch.Tensor


@dataclasses.dataclass
class _Outer:
    """Dataclass holding any value."""

    value: object


@dataclasses.dataclass
class _Empty:
    """Dataclass without fields."""


class _NotADataClass:
    """Object that is neither a container nor a dataclass."""

    t1: torch.Tensor

    def __init__(self, t1: torch.Tensor) -> None:
        self.t1 = t1
        return


def get_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Get inputs for tests."""
    return torch.tensor(1.0), torch.tensor(2.0)


def get_apply_object_data() -> list[_BaseTestClass]:
    """Get test data for the apply function."""
    inputs = get_inputs()
    return [
        DataClass(*inputs),
        SlottedDataClass(*inputs),
        FrozenDataClass(*inputs),
        DataClassWithInit(*inputs),
    ]


def _times_two(x: torch.Tensor) -> torch.Tensor:
    return 2 * x


def test_apply_fails() -> None:
    """Test apply fails when the wrong type is given."""
    expected_type = torch.Tensor
    tuple_data = (torch.tensor(1.0), [1, 2])
    dict_data = {'list': tuple_data}

    # fail because it expects torch.Tensors and not int
    with pytest.raises(exceptions.FuncNotApplicableError):
        apply(dict_data, expected_type=expected_type, func=_times_two)


def test_apply_rejects_other_objects() -> None:
    """Test an object with attributes that is not a dataclass raises."""
    with pytest.raises(exceptions.FuncNotApplicableError):
        apply(
            _NotADataClass(torch.tensor(1.0)),
            expected_type=torch.Tensor,
            func=_times_two,
        )


def test_apply_containers() -> None:
    """Test apply works as expected."""
    expected_type = torch.Tensor
    new_tuple_data = [
        torch.tensor(1.0),
        _TorchTuple(torch.tensor(1.0), torch.tensor(2.0)),
    ]
    new_dict_data = {'list': new_tuple_data}

    out = apply(new_dict_data, expected_type=expected_type, func=_times_two)

    expected = {
        'list': [
            torch.tensor(2.0),
            _TorchTuple(torch.tensor(2.0), torch.tensor(4.0)),
        ]
    }
    assert out == expected


@pytest.mark.parametrize('obj', get_apply_object_data())
def test_apply(obj: _BaseTestClass) -> None:
    """Test apply works on a variety of classes."""
    new_obj = apply(obj, torch.Tensor, _times_two)

    assert new_obj.t1 == 2.0
    assert new_obj.t2 == 4.0
    assert new_obj is not obj


@pytest.mark.parametrize('inner_cls', [_Inner, _SlottedInner, _FrozenInner])
def test_apply_nested_dataclass(inner_cls: type) -> None:
    """Test a dataclass inside a dataclass is traversed and copied."""
    inner = inner_cls(torch.tensor(1.0))

    new_obj = apply(_Outer(inner), torch.Tensor, _times_two)

    assert new_obj.value.t1 == 2.0
    assert inner.t1 == 1.0


def test_apply_dataclass_in_list_in_dataclass() -> None:
    """Test a dataclass is reached through a container."""
    new_obj = apply(
        _Outer([_Inner(torch.tensor(1.0))]), torch.Tensor, _times_two
    )

    assert new_obj.value[0].t1 == 2.0


def test_apply_dataclass_without_fields() -> None:
    """Test a dataclass with nothing to traverse is simply copied."""
    obj = _Empty()

    new_obj = apply(_Outer(obj), torch.Tensor, _times_two)

    assert isinstance(new_obj.value, _Empty)


def test_apply_keeps_none() -> None:
    """Test an attribute set to None stays None."""
    new_obj = apply(_Outer(None), torch.Tensor, _times_two)

    assert new_obj.value is None


def test_apply_cpu_detach_nested_dataclass() -> None:
    """Test tensors inside a nested dataclass are detached."""
    tensor = torch.tensor(1.0, requires_grad=True)

    new_obj = apply_cpu_detach(_Outer(_Inner(tensor)))

    assert not new_obj.value.t1.requires_grad


def test_apply_dataclass_init_false() -> None:
    """Test field excluded from constructor is transformed, not recomputed."""
    t = torch.tensor(1.0)
    obj = InitFalseDataClass(t)

    new_obj = apply(obj, torch.Tensor, _times_two)

    assert new_obj.t1.item() == 2.0
    assert new_obj.t2.item() == 4.0


def test_apply_dataclass_field_assigned_after_construction() -> None:
    """Test field assigned after construction keeps its value."""
    t = torch.tensor(1.0)
    obj = InitFalseDataClass(t)
    obj.t2 = torch.tensor(5.0)

    new_obj = apply(obj, torch.Tensor, _times_two)

    assert new_obj.t1.item() == 2.0
    assert new_obj.t2.item() == 10.0


def test_apply_dataclass_construction_machinery_agreement() -> None:
    """Test classes differing only in construction machinery agree."""
    t = torch.tensor(1.0)
    init_false_obj = InitFalseDataClass(t)
    init_var_obj = InitVarDataClass(t)

    new_init_false = apply(init_false_obj, torch.Tensor, _times_two)
    new_init_var = apply(init_var_obj, torch.Tensor, _times_two)

    assert new_init_false.t1.item() == 2.0
    assert new_init_var.t1.item() == 2.0
    assert new_init_false.t2.item() == 4.0
    assert new_init_var.t2.item() == 4.0
    assert new_init_false.t1 == new_init_var.t1
    assert new_init_false.t2 == new_init_var.t2


def test_apply_slot_declared_on_base_class() -> None:
    """Test a slot declared on a base class is transformed."""
    t = torch.tensor(1.0)
    extra = torch.tensor(3.0)
    obj = _DerivedWithBaseSlot(t)
    obj.extra = extra

    new_obj = apply(obj, torch.Tensor, _times_two)

    assert new_obj.t1.item() == 2.0
    assert new_obj.extra.item() == 6.0


def test_apply_manual_slot_entry_not_a_field() -> None:
    """Test a manual slot entry that is not a field is transformed."""
    a = torch.tensor(1.0)
    cache = torch.tensor(4.0)
    obj = _DataClassWithManualSlot(a)
    obj.cache = cache

    new_obj = apply(obj, torch.Tensor, _times_two)

    assert new_obj.a.item() == 2.0
    assert new_obj.cache.item() == 8.0


def test_recursive_to() -> None:
    """Test ``recursive_to`` works as expected."""
    list_data = _TorchLikeTuple(
        torch.tensor(1.0), [torch.tensor(1.0), torch.tensor(2.0)]
    )
    device = torch.device('meta')

    list_data = apply_to(list_data, device=device)

    assert list_data[0].device == device
    assert list_data[1][0].device == device
    assert list_data[1][1].device == device


def test_apply_cpu_detach() -> None:
    """Test apply_cpu_detach."""
    t = torch.tensor([1.0], requires_grad=True)
    obj = {'a': t}

    new_obj = apply_cpu_detach(obj)

    assert not new_obj['a'].requires_grad
    assert new_obj['a'].item() == 1.0


def test_apply_class_object_raises_error() -> None:
    """Test apply raises on a class object."""
    cls = DataClass

    with pytest.raises(exceptions.FuncNotApplicableError):
        apply(cls, torch.Tensor, _times_two)


def test_apply_to_collated_batch() -> None:
    """Test a collated batch shape with strings works end to end."""
    batch = (torch.zeros(2, 3), ['sample_a', 'sample_b'])
    device = torch.device('cpu')

    out = apply_to(batch, device=device)

    assert torch.equal(out[0], torch.zeros(2, 3))
    assert out[1] == ['sample_a', 'sample_b']


def test_apply_strings_and_bytes_passthrough() -> None:
    """Test strings and bytes pass through wherever they sit."""
    data = {
        'tensor': torch.tensor(1.0),
        'bare_str': 'sample_string',
        'bare_bytes': b'sample_bytes',
        'list_str': ['a', 'b'],
        'named_tuple': _NamedTupleWithString('foo'),
    }

    out = apply(data, torch.Tensor, _times_two)

    assert out['tensor'].item() == 2.0
    assert out['bare_str'] == 'sample_string'
    assert out['bare_bytes'] == b'sample_bytes'
    assert out['list_str'] == ['a', 'b']
    assert out['named_tuple'] == _NamedTupleWithString('foo')


def test_apply_preserves_string_identity() -> None:
    """Test a str leaf comes back as the same object rather than a copy."""
    leaf = 'sample_string'
    container = [leaf]

    out = apply(container, torch.Tensor, _times_two)

    assert out[0] is leaf


def test_apply_unsupported_types_still_raise() -> None:
    """Test apply raises for a set and a plain object."""
    set_leaf = [{1, 2}]
    obj_leaf = [object()]

    with pytest.raises(exceptions.FuncNotApplicableError) as exc_set:
        apply(set_leaf, torch.Tensor, _times_two)

    with pytest.raises(exceptions.FuncNotApplicableError) as exc_obj:
        apply(obj_leaf, torch.Tensor, _times_two)

    assert exc_set.type is exceptions.FuncNotApplicableError
    assert exc_obj.type is exceptions.FuncNotApplicableError


def test_passthrough_types_constant() -> None:
    """Test _PASSTHROUGH_TYPES constant contains str, bytes and None."""
    expected = (str, bytes, type(None))

    actual = _PASSTHROUGH_TYPES

    assert actual == expected


def test_apply_expected_type_str() -> None:
    """Test calling apply with expected_type=str transforms it."""
    s = 'hello'

    out = apply(s, str, str.upper)

    assert out == 'HELLO'

"""Tests for the "apply_ops" module."""

from typing import NamedTuple

import torch

import pytest

from drytorch.core import exceptions
from drytorch.utils.apply_ops import apply_to, recursive_apply


class _TorchTuple(NamedTuple):
    one: torch.Tensor
    two: torch.Tensor


class _TorchLikeTuple(NamedTuple):
    tensor: torch.Tensor
    tensor_lst: list[torch.Tensor]


def test_recursive_apply() -> None:
    """Test works when expected to."""
    expected_type = torch.Tensor
    tuple_data = (torch.tensor(1.0), [1, 2])
    dict_data = {'list': tuple_data}

    def _times_two(x: torch.Tensor) -> torch.Tensor:
        return 2 * x

    # fail because it expects torch.Tensors and not int
    with pytest.raises(exceptions.FuncNotApplicableError):
        recursive_apply(
            obj=dict_data, expected_type=expected_type, func=_times_two
        )

    new_tuple_data = [
        torch.tensor(1.0),
        _TorchTuple(torch.tensor(1.0), torch.tensor(2.0)),
    ]
    new_dict_data = {'list': new_tuple_data}
    out = recursive_apply(
        obj=new_dict_data, expected_type=expected_type, func=_times_two
    )
    expected = {
        'list': [
            torch.tensor(2.0),
            _TorchTuple(torch.tensor(2.0), torch.tensor(4.0)),
        ]
    }
    assert out == expected


def test_recursive_to() -> None:
    """Test ``recursive_to`` works as expected."""
    list_data = _TorchLikeTuple(
        torch.tensor(1.0), [torch.tensor(1.0), torch.tensor(2.0)]
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    list_data = apply_to(list_data, device=device)
    assert list_data[0].device == device
    assert list_data[1][0].device == device

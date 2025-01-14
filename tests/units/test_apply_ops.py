"""Tests for the apply_ops module"""

from typing import NamedTuple
import pytest
import torch

from src.dry_torch.apply_ops import recursive_apply
from src.dry_torch.apply_ops import apply_to


class TorchTuple(NamedTuple):
    """Example input."""
    one: torch.Tensor
    two: torch.Tensor


class TorchLikeTuple(NamedTuple):
    """Example input."""
    tensor: torch.Tensor
    tensor_lst: list[torch.Tensor]


def test_recursive_apply() -> None:
    expected_type = torch.Tensor
    tuple_data = (torch.tensor(1.), [1, 2])
    dict_data = {'list': tuple_data}

    def _times_two(x: torch.Tensor) -> torch.Tensor:
        return 2 * x

    # fail because it expects torch.Tensors and not int
    with pytest.raises(TypeError):
        recursive_apply(obj=dict_data,
                        expected_type=expected_type,
                        func=_times_two)

    new_tuple_data = [torch.tensor(1.),
                      TorchTuple(torch.tensor(1.), torch.tensor(2.))]
    new_dict_data = {'list': new_tuple_data}
    out = recursive_apply(obj=new_dict_data,
                          expected_type=expected_type,
                          func=_times_two)
    expected = {'list': [torch.tensor(2.),
                         TorchTuple(torch.tensor(2.), torch.tensor(4.))]}
    assert out == expected


def test_recursive_to() -> None:
    list_data = TorchLikeTuple(torch.tensor(1.),
                               [torch.tensor(1.), torch.tensor(2.)])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    list_data = apply_to(list_data, device=device)
    assert list_data[0].device == device
    assert list_data[1][0].device == device

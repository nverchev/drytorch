import pytest
import torch
from dry_torch.recursive_ops import recursive_apply
from dry_torch.recursive_ops import recursive_to


def test_recursive_apply() -> None:
    expected_type = torch.Tensor
    tuple_data = (torch.tensor(1.), [1, 2])
    dict_data = {'list': tuple_data}

    def times_two(x: torch.Tensor) -> torch.Tensor:
        return 2 * x

    # fail because it expects torch.Tensors and not int
    with pytest.raises(TypeError):
        recursive_apply(struc=dict_data,
                        expected_type=expected_type,
                        func=times_two)

    new_tuple_data = [torch.tensor(1.), (torch.tensor(1.), torch.tensor(2.))]
    new_dict_data = {'list': new_tuple_data}
    out = recursive_apply(struc=new_dict_data,
                          expected_type=expected_type,
                          func=times_two)
    expected = {'list': [torch.tensor(2.),
                         (torch.tensor(2.), torch.tensor(4.))]}
    assert out == expected

    # check annotations
    _out2 = recursive_apply(struc=torch.tensor(1.),
                            expected_type=expected_type,
                            func=str)
    _out3 = recursive_apply(struc=torch.tensor(1.),
                            expected_type=expected_type,
                            func=str)


def test_recursive_to() -> None:
    list_data = (torch.tensor(1.), [torch.tensor(1.), torch.tensor(2.)])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    list_data = recursive_to(list_data, device=device)
    assert list_data[0].device == device
    assert list_data[1][0].device == device

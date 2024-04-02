import pytest
import torch
from custom_trainer.dict_list import DictList, ListKeyError


def test_DictList() -> None:
    tensor_dict_list: DictList[str, torch.Tensor] = DictList()
    input_dict_list = [{'list1': torch.ones(2, 2)} for _ in range(2)]
    tensor_dict_list.extend(input_dict_list)
    assert tensor_dict_list == DictList(input_dict_list)
    tensor_dict_list.append({'list1': torch.ones(2, 2)})
    # this prevents dictionaries from having different keys
    with pytest.raises(ListKeyError):
        tensor_dict_list.extend([{'list2': torch.ones(2, 2) for _ in range(5)}])
    with pytest.raises(ListKeyError):
        tensor_dict_list[0] = {'list2': torch.ones(2, 2)}
    with pytest.raises(ListKeyError):
        tensor_dict_list.insert(0, {'list2': torch.ones(2, 2)})
    tensor_dict_list2 = tensor_dict_list.copy()
    tensor_dict_list.extend(tensor_dict_list)
    assert tensor_dict_list == tensor_dict_list2 + tensor_dict_list2

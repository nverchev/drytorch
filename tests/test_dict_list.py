import pytest
import torch
from dry_torch import TorchDictList
from dry_torch.structures import DictList
from dry_torch.exceptions import KeysAlreadySetError
from dry_torch.exceptions import DifferentBatchSizeError
from dry_torch.data_types import Tensors


def test_DictList() -> None:
    dict_list: DictList[str, torch.Tensor] = DictList()
    input_dict_list = [{'list1': torch.ones(2, 2)} for _ in range(2)]

    # test __init__ and extend
    dict_list.extend(input_dict_list)
    assert torch.allclose(dict_list[0]['list1'], torch.ones(2, 2))
    dict_list[1] = {'list1': torch.ones(2, 2)}

    # test append and __getitem__
    assert torch.allclose(dict_list[1]['list1'],
                          DictList(input_dict_list)[1]['list1'])
    dict_list.append({'list1': torch.ones(2, 2)})

    # test KeysAlreadySet
    with pytest.raises(KeysAlreadySetError):
        dict_list.extend([{'list2': torch.ones(2, 2) for _ in range(5)}])
    with pytest.raises(KeysAlreadySetError):
        dict_list[0] = {'list2': torch.ones(2, 2)}
    with pytest.raises(KeysAlreadySetError):
        dict_list.insert(0, {'list2': torch.ones(2, 2)})

    # test __add__ and extend
    tensor_dict_list2 = dict_list.copy()
    dict_list.extend(dict_list)
    assert dict_list == tensor_dict_list2 + tensor_dict_list2

    # test pop
    assert torch.allclose(dict_list.pop()['list1'], torch.ones(2, 2))

    # test get
    assert dict_list.to_dict()['list1'] == dict_list.get('list1')
    assert dict_list.get('list3') == [None, None, None, None, None]
    assert list(dict_list.keys()) == ['list1']


def test_TorchDictList() -> None:
    input_dict_torch: dict[str, Tensors] = {
        'tensor1': torch.ones(2),
        'list_tensor': [torch.zeros(2)]
    }

    # test enlist
    expected_result = 2 * [(torch.tensor(1), (torch.tensor(0), ))]
    assert TorchDictList._enlist(input_dict_torch.values()) == expected_result

    # test DifferentBatchSizeError
    wrong_input_dict_torch: dict[str, Tensors] = {
        'tensor1': torch.ones(2, 2),
        'list_tensor': [torch.zeros(1, 2)]
    }
    with pytest.raises(DifferentBatchSizeError):
        TorchDictList.from_batch(wrong_input_dict_torch)

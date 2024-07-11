import pytest
import torch
from typing import NamedTuple, Sequence

from dry_torch import TorchDictList
from dry_torch.structures import DictList
from dry_torch.exceptions import KeysAlreadySetError
from dry_torch.exceptions import DifferentBatchSizeError


class BatchTuple(NamedTuple):
    output1: torch.Tensor
    output2: torch.Tensor


class ListedBatchTuple(NamedTuple):
    output1: list[torch.Tensor]
    output2: list[torch.Tensor]


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


tuple_list_type = Sequence[
    tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]
]


def check_equal(tuple_list: tuple_list_type,
                expected_result: tuple_list_type) -> None:
    for t1, t2 in zip(tuple_list, expected_result):
        for t11, t22 in zip(t1, t2):
            if isinstance(t11, torch.Tensor):
                assert isinstance(t22, torch.Tensor)
                assert torch.allclose(t11, t22)
            else:
                for t111, t222 in zip(t11, t22):
                    assert isinstance(t111, torch.Tensor)
                    assert isinstance(t222, torch.Tensor)
                    assert torch.allclose(t111, t222)
    return


def test_TorchDictList() -> None:
    batch_tuple = BatchTuple(torch.ones(2, 2), torch.zeros(2, 2))
    expected_result: tuple_list_type = 2 * [
        (torch.tensor(1.), torch.tensor(0.))
    ]
    tuple_list = TorchDictList.from_batch(batch_tuple)._tuple_list

    check_equal(tuple_list, expected_result)

    listed_batch_tuple = ListedBatchTuple(
        [torch.ones(2, 2)], [torch.zeros(2, 2)]
    )
    expected_result = 2 * [((torch.ones(2),), (torch.zeros(2),))]
    tuple_list = TorchDictList.from_batch(listed_batch_tuple)._tuple_list

    check_equal(tuple_list, expected_result)
    # test DifferentBatchSizeError
    wrong_batch_tuple = ListedBatchTuple(
        [torch.ones(2, 2)], [torch.zeros(1, 2)]
    )
    with pytest.raises(DifferentBatchSizeError):
        TorchDictList.from_batch(wrong_batch_tuple)

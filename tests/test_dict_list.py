import numpy as np
import numpy.typing as npt
import pytest
import torch
from typing import Sequence
import dataclasses

from dry_torch import NumpyDictList
from dry_torch import structures
from dry_torch import exceptions


@dataclasses.dataclass(slots=True)
class BatchOutput:
    output1: torch.Tensor
    output2: list[torch.Tensor]

    def to_dict(self) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return {'output1': self.output1, 'output2': self.output2}


@dataclasses.dataclass(slots=True)
class WrongTypeBatch:
    output1: int
    output2: list[torch.Tensor]

    def to_dict(self) -> dict[str, int | list[torch.Tensor]]:
        return {'output1': self.output1, 'output2': self.output2}


@dataclasses.dataclass(slots=True)
class WrongListedTypeBatch:
    output1: torch.Tensor
    output2: list[int]

    def to_dict(self) -> dict[str, torch.Tensor | list[int]]:
        return {'output1': self.output1, 'output2': self.output2}


@dataclasses.dataclass(slots=True)
class NoToDictBatch:
    output1: torch.Tensor
    output2: list[torch.Tensor]


def test_DictList() -> None:
    dict_list: structures.DictList[str, torch.Tensor] = structures.DictList()
    input_dict_list = [{'list1': torch.ones(2, 2)} for _ in range(2)]

    # test __init__ and extend
    dict_list.extend(input_dict_list)
    assert torch.allclose(dict_list[0]['list1'], torch.ones(2, 2))
    dict_list[1] = {'list1': torch.ones(2, 2)}

    # test append and __getitem__
    assert torch.allclose(dict_list[1]['list1'],
                          structures.DictList(input_dict_list)[1]['list1'])
    dict_list.append({'list1': torch.ones(2, 2)})

    # test KeysAlreadySet
    with pytest.raises(exceptions.KeysAlreadySetError):
        dict_list.extend([{'list2': torch.ones(2, 2) for _ in range(5)}])
    with pytest.raises(exceptions.KeysAlreadySetError):
        dict_list[0] = {'list2': torch.ones(2, 2)}
    with pytest.raises(exceptions.KeysAlreadySetError):
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
    tuple[npt.NDArray | tuple[npt.NDArray, ...], ...]
]


def check_equal(tuple_list: tuple_list_type,
                expected_result: tuple_list_type) -> None:
    for t1, t2 in zip(tuple_list, expected_result):
        for t11, t22 in zip(t1, t2):
            if isinstance(t11, np.ndarray):
                assert np.allclose(t11, t22)
            else:
                for t111, t222 in zip(t11, t22):
                    assert np.allclose(t111, t222)
    return


def test_NumpyDictList() -> None:
    batch_batch = BatchOutput(torch.ones(2, 2), [torch.zeros(2, 2)])
    expected_result: tuple_list_type = 2 * [(np.array(1), (np.array(0),))]
    tuple_list = NumpyDictList.from_batch(batch_batch)._tuple_list

    check_equal(tuple_list, expected_result)
    # test DifferentBatchSizeError
    wrong_batch_tuple = BatchOutput(torch.ones(2, 2), [torch.zeros(1, 2)])
    with pytest.raises(exceptions.DifferentBatchSizeError):
        NumpyDictList.from_batch(wrong_batch_tuple)

    # test NotATensorError
    wrong_type_batch = WrongTypeBatch(2, [torch.zeros(2, 2)])
    with pytest.raises(exceptions.NotATensorError):
        NumpyDictList.from_batch(wrong_type_batch)
    wrong_listed_type_batch = WrongListedTypeBatch(torch.ones(2, 2), [2, 2])
    with pytest.raises(exceptions.NotATensorError):
        NumpyDictList.from_batch(wrong_listed_type_batch)

    # test NoToDictMethodError
    no_to_dict_batch = NoToDictBatch(torch.ones(2, 2), [torch.zeros(2, 2)])
    with pytest.raises(exceptions.NoToDictMethodError):
        NumpyDictList.from_batch(no_to_dict_batch)

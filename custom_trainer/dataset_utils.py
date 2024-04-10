from typing import TypeVar, Optional, Literal, overload

import torch
from torch.utils.data import Dataset, DataLoader, StackDataset

Inputs = TypeVar('Inputs')
Targets = TypeVar('Targets')
Data = tuple[Inputs, Targets]
IndexData = tuple[Data, tuple[torch.LongTensor,]]


@overload
def get_indexed_loader(dataset: None, batch_size: int, partition: Literal['train', 'val', 'test'] = ...) -> None:
    ...


@overload
def get_indexed_loader(dataset: Dataset[Data], batch_size: int, partition: Literal['train', 'val', 'test'] = ...
                       ) -> DataLoader[IndexData]:
    ...


def get_indexed_loader(dataset: Optional[Dataset[Data]],
                       batch_size: int,
                       partition: Literal['train', 'val', 'test'] = 'train'
                       ) -> Optional[DataLoader[IndexData]]:
    if dataset is None:
        return None
    indexed_dataset = StackDataset(dataset, IndexDataset())  # add indices
    pin_memory: bool = torch.cuda.is_available()
    drop_last: bool = True if partition == 'train' else False
    shuffle: bool = True if partition == 'train' else False
    return DataLoader(indexed_dataset, drop_last=drop_last, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory)


class IndexDataset(Dataset):
    """
    This class is used to create a dataset that can be used in a DataLoader.
    """

    def __getitem__(self, index) -> torch.LongTensor:
        return torch.LongTensor(index)

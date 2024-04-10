from typing import TypeVar, Optional, Literal, overload, Generic

import torch
from torch.utils.data import Dataset, DataLoader, StackDataset

Inputs = TypeVar('Inputs')
Targets = TypeVar('Targets')
Data = tuple[Inputs, Targets]
IndexData = tuple[Data, tuple[torch.LongTensor,]]

DatasetName = Literal['train', 'val', 'test']


class IndexDataset(Dataset):
    """
    This class is used to create a dataset that can be used in a DataLoader.
    """

    def __getitem__(self, index) -> torch.LongTensor:
        return torch.LongTensor(index)


class DataManager(Generic[Inputs, Targets]):

    def __init__(self,
                 train_dataset: Dataset[Data],
                 val_dataset: Optional[Dataset[Data]],
                 test_dataset: Optional[Dataset[Data]],
                 batch_size: int) -> None:
        self.loaders: dict[DatasetName, DataLoader[IndexData]] = {}
        self.batch_size = batch_size
        dataset_names: tuple[DatasetName, ...] = ('train', 'val', 'test')
        for dataset, dataset_name in zip((train_dataset, val_dataset, test_dataset), dataset_names):
            self.set_loader(dataset, dataset_name)

    def set_loader(self,
                   dataset: Optional[Dataset[Data]],
                   dataset_name: DatasetName = 'train'
                   ):
        if dataset is None:
            return None
        indexed_dataset = StackDataset(dataset, IndexDataset())  # add indices
        pin_memory: bool = torch.cuda.is_available()
        drop_last: bool = True if dataset_name == 'train' else False
        shuffle: bool = True if dataset_name == 'train' else False
        self.loaders[dataset_name] = DataLoader(indexed_dataset, drop_last=drop_last, batch_size=self.batch_size,
                                                shuffle=shuffle, pin_memory=pin_memory)

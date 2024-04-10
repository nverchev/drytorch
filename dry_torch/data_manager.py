from typing import TypeVar, Optional, Literal, Generic

import torch
from torch.utils.data import Dataset, DataLoader, StackDataset

Data_co = TypeVar('Data_co', covariant=True)
IndexData = tuple[Data_co, tuple[torch.LongTensor,]]
DatasetName = Literal['train', 'val', 'test']


class IndexDataset(Dataset):
    """
    Return the index sampled by a DataLoader.
    """
    def __getitem__(self, index) -> torch.LongTensor:
        return torch.LongTensor(index)


class IndexedDataLoader(Generic[Data_co]):
    """
    Construct a DataLoader that keep track of the sampled indices for the datasets.

    Args:
        train_dataset: the training dataset
        val_dataset: the validation dataset, optional
        test_dataset: the test dataset, optional dataset
        batch_size: the batch size
        max_num_batch: the maximum number of batches for datasets that do not have a predetermined length.
    Attributes:
        loaders: a dictionary that stores the loaders for each dataset.
        max_num_batch: the maximum number of batches for datasets that do not have a predetermined length.
    """

    def __init__(self,
                 train_dataset: Dataset[Data_co],
                 val_dataset: Optional[Dataset[Data_co]],
                 test_dataset: Optional[Dataset[Data_co]],
                 batch_size: int,
                 max_num_batch: int = 1024,
                 ) -> None:

        self.loaders: dict[DatasetName, DataLoader[IndexData]] = {}
        self.max_num_batch = max_num_batch
        dataset_names: tuple[DatasetName, ...] = ('train', 'val', 'test')
        for dataset, dataset_name in zip((train_dataset, val_dataset, test_dataset), dataset_names):
            self._set_indexed_loader(dataset, dataset_name, batch_size)

    def _set_indexed_loader(self,
                            dataset: Optional[Dataset[Data_co]],
                            dataset_name: DatasetName,
                            batch_size: int
                            ):
        if dataset is None:
            return None
        indexed_dataset = StackDataset(dataset, IndexDataset())  # add indices
        pin_memory: bool = torch.cuda.is_available()
        drop_last: bool = True if dataset_name == 'train' else False
        shuffle: bool = True if dataset_name == 'train' else False
        self.loaders[dataset_name] = DataLoader(indexed_dataset, drop_last=drop_last, batch_size=batch_size,
                                                shuffle=shuffle, pin_memory=pin_memory)

from typing import Optional, TypeVar, TypeAlias
import torch
from torch.utils import data
from dry_torch import exceptions
from dry_torch import protocols
from dry_torch import data_types

_Input_co = TypeVar('_Input_co', bound=data_types.InputType, covariant=True)
_Target_co = TypeVar('_Target_co', bound=data_types.InputType, covariant=True)


class Loaders(protocols.LoadersProtocol[_Input_co, _Target_co]):
    """
    A container for the data _static_loaders.
    Args:
        train_dataset: dataset used for training
        val_dataset: dataset used for validation
        test_dataset: dataset used for testing
        batch_size: the batch size.
        runtime_build: whether to build the data _static_loaders at runtime, with
        settings based on the inference model.
                       Defaults to False.
    """

    def __init__(
            self,
            batch_size: int,
            train_dataset: (
                    Optional[data.Dataset[tuple[_Input_co, _Target_co]]]
            ) = None,
            val_dataset: (
                    Optional[data.Dataset[tuple[_Input_co, _Target_co]]]
            ) = None,
            test_dataset: (
                    Optional[data.Dataset[tuple[_Input_co, _Target_co]]]
            ) = None,
            runtime_build: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self._runtime_build = runtime_build
        datasets = (train_dataset, val_dataset, test_dataset)
        self.datasets: (
            dict[data_types.Split, data.Dataset[tuple[_Input_co, _Target_co]]]
        ) = {}
        for split, dataset in zip(data_types.Split, datasets):
            if dataset is not None:
                self.datasets[split] = dataset

        self.datasets_length: data_types.PartitionsLength = {}
        for split, dataset in self.datasets.items():
            if hasattr(dataset, '__len__'):
                self.datasets_length[split] = dataset.__len__()
            else:
                raise exceptions.PartitionNotFoundError(split.name)
        self._static_loaders: data_types.LoadersDict = {}

    def get_loader(
            self,
            partition: data_types.Split,
    ) -> data.DataLoader[tuple[_Input_co, _Target_co]]:

        if partition not in self.datasets:
            raise exceptions.PartitionNotFoundError(partition.name)
        if self._runtime_build:
            return self._get_loader(self.datasets[partition],
                                    self.batch_size,
                                    torch.is_inference_mode_enabled())
        if partition not in self._static_loaders:
            inference: bool = partition != data_types.Split.TRAIN
            loader = self._get_loader(self.datasets[partition],
                                      self.batch_size,
                                      inference)
            self._static_loaders[partition] = loader
        return self._static_loaders[partition]

    @staticmethod
    def _get_loader(
            dataset: data.Dataset[tuple[_Input_co, _Target_co]],
            batch_size: int,
            inference: bool,
    ) -> data.DataLoader[tuple[_Input_co, _Target_co]]:
        pin_memory: bool = torch.cuda.is_available()
        drop_last: bool = not inference
        shuffle: bool = not inference
        return data.DataLoader(dataset,
                               batch_size=batch_size,
                               drop_last=drop_last,
                               shuffle=shuffle,
                               pin_memory=pin_memory)

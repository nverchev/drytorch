import dataclasses
from typing import NamedTuple

import torch
from torch.utils import data


class TorchTuple(NamedTuple):
    input: torch.Tensor


@dataclasses.dataclass()
class TorchData:
    output: torch.Tensor
    output2: tuple[torch.Tensor, ...] = (torch.empty(0),)


class IdentityDataset(data.Dataset[tuple[TorchTuple, torch.Tensor]]):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index: int) -> tuple[TorchTuple, torch.Tensor]:
        x = torch.FloatTensor([index]) / len(self)
        return TorchTuple(x), x

    def __len__(self) -> int:
        return 1600


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs: TorchTuple) -> TorchData:
        return TorchData(self.linear(inputs.input))

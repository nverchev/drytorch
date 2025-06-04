"""Module with simple class definitions for testing."""

import dataclasses
from typing import NamedTuple

import torch
from torch.utils import data


class TorchTuple(NamedTuple):
    """Simple input class for a neural network model."""
    input: torch.Tensor


@dataclasses.dataclass()
class TorchData:
    """Simple output class for a neural network model."""
    output: torch.Tensor
    output2: tuple[torch.Tensor, ...] = (torch.empty(0),)


class IdentityDataset(data.Dataset[tuple[TorchTuple, torch.Tensor]]):
    """Simple dataset class to learn the identity function."""

    def __init__(self):
        super().__init__()

    def __getitem__(self, index: int) -> tuple[TorchTuple, torch.Tensor]:
        x = torch.FloatTensor([index]) / len(self)
        return TorchTuple(x), x

    def __len__(self) -> int:
        return 64


class Linear(torch.nn.Module):
    """
    Wrapper around a linear model with structured input / output.

    Attributes:
        linear: PyTorch linear layer.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: input dimension.
            out_features: output dimension.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs: TorchTuple) -> TorchData:
        """
        Performs a forward pass on the input.
        Args:
            inputs: structured input

        Returns:
            structured output.
        """
        return TorchData(self.linear(inputs.input))

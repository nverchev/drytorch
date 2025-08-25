"""Module with simple class definitions for testing."""

import dataclasses

from typing import NamedTuple

import torch

from torch.utils import data
from typing_extensions import override


class TorchTuple(NamedTuple):
    """Simple input class for a neural network model."""

    input: torch.Tensor


@dataclasses.dataclass(frozen=True)
class TorchData:
    """Simple output class for a neural network model."""

    output: torch.Tensor
    output2: tuple[torch.Tensor, ...] = (torch.empty(0),)


class IdentityDataset(data.Dataset[tuple[TorchTuple, torch.Tensor]]):
    """Simple dataset class to learn the identity function."""

    def __init__(self, len_epoch=64):
        """Constructor.

        Args:
            len_epoch: how many samples to generate each epoch.
        """
        super().__init__()
        self.len_epoch = len_epoch

    @override
    def __getitem__(self, index: int) -> tuple[TorchTuple, torch.Tensor]:
        x = torch.FloatTensor([index]) / len(self)
        return TorchTuple(x), x

    def __len__(self) -> int:
        """Number of samples."""
        return self.len_epoch


class Linear(torch.nn.Module):
    """Wrapper around a linear model with structured input / output.

    Attributes:
        linear: PyTorch linear layer.
    """

    def __init__(self, in_features: int, out_features: int):
        """Constructor.

        Args:
            in_features: input dimension.
            out_features: output dimension.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs: TorchTuple) -> TorchData:
        """Constructor.

        Args:
            inputs: structured input

        Returns:
            structured output.
        """
        return TorchData(self.linear(inputs.input))

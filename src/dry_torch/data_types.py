import enum
from typing import TypeAlias
import pathlib
import pandas as pd
import torch
from torch.utils import data


class Split(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


Tensors: TypeAlias = (
        torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
)
InputType: TypeAlias = Tensors | dict[str, Tensors]
TargetType: TypeAlias = Tensors | dict[str, Tensors]
OutputType: TypeAlias = Tensors | dict[str, Tensors]
PartitionsLength: TypeAlias = dict[Split, int]
LoadersDict: TypeAlias = dict[Split, data.DataLoader]
LogsDict: TypeAlias = dict[Split, pd.DataFrame]
PathDict: TypeAlias = dict[Split, pathlib.Path]

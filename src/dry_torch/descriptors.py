import enum
import pathlib
from typing import TypedDict, Iterator, TypeAlias

import pandas as pd
from torch.nn import Parameter


class Split(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


class OptParams(TypedDict):
    params: Iterator[Parameter]
    lr: float


class StatePath(TypedDict):
    state: pathlib.Path
    optimizer: pathlib.Path


PartitionsLength: TypeAlias = dict[Split, int]
LogsDict: TypeAlias = dict[Split, pd.DataFrame]
PathDict: TypeAlias = dict[Split, pathlib.Path]

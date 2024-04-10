from typing import TypeVar, Protocol, TypedDict, Iterator

import torch
from torch.nn.parameter import Parameter

ModuleInput = TypeVar('ModuleInput', contravariant=True)
ModuleOutput = TypeVar('ModuleOutput', covariant=True)


class ModuleProtocol(Protocol[ModuleInput, ModuleOutput]):

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        ...

    def __call__(self, inputs: ModuleInput) -> ModuleOutput:
        ...


class TypedModule(torch.nn.Module, ModuleProtocol[ModuleInput, ModuleOutput]):
    """torch.nn.Module with Generic notation."""

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        return super().forward(inputs)

    def __call__(self, inputs: ModuleInput) -> ModuleOutput:
        return super().__call__(inputs)


class MetricsProtocol(Protocol):

    # noinspection PyPropertyDefinition
    @property
    def metrics(self) -> dict[str, torch.Tensor]:
        ...


class LossAndMetricsProtocol(Protocol):
    criterion: torch.FloatTensor

    # noinspection PyPropertyDefinition
    @property
    def metrics(self) -> dict[str, torch.Tensor]:
        ...


class OptParams(TypedDict):
    params: Iterator[Parameter]
    lr: float

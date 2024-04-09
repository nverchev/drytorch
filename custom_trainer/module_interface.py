from typing import TypeVar, Protocol

import torch

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


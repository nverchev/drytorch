from __future__ import annotations

import torch
from torch import Tensor

from custom_trainer.exceptions import NotATensorError


class Metrics:
    __slots__: tuple[str, ...] = tuple()

    @classmethod
    def __new__(cls, *args, **kwargs) -> Metrics:
        instance = super().__new__(cls)

        # Compatible with non-slotted classes.
        if getattr(instance, '__dict__', False):
            instance.__slots__ = tuple(kwargs.keys())
        else:
            unused_slots = set(instance.__slots__) - set(kwargs.keys())
            if unused_slots:
                raise AttributeError(f"Slotted attribute not initialized: {unused_slots}")
        return instance

    def __init__(self, /, **kwargs: torch.Tensor) -> None:
        for name, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                raise NotATensorError(value, name)
            setattr(self, name, value)

    @property
    def metrics(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in self.__slots__ if name != 'criterion'}

    def clear(self) -> None:
        for name in self.__slots__:
            setattr(self, name, torch.tensor(0.0))


class LossAndMetrics(Metrics):
    __slots__: tuple[str, ...] = ('criterion',)

    def __init__(self, criterion: torch.Tensor, **kwargs: torch.Tensor) -> None:
        self.criterion = criterion
        super().__init__(**kwargs)

"""Classes that contain event information for logging purposes."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable
from typing import Any, Optional, Mapping
import pathlib


class Event(metaclass=abc.ABCMeta):
    """Class for logging events"""
    _auto_publish: Callable[[Event], None] = lambda self: None

    def __post_init__(self):
        self.__class__._auto_publish(self)

    @classmethod
    def set_auto_publish(cls, func: Callable[[Event], None]) -> None:
        """Specify how to notify subscribers upon creation."""
        cls._auto_publish = func
        return


@dataclasses.dataclass
class StartExperiment(Event):
    exp_name: str
    exp_dir: pathlib.Path


@dataclasses.dataclass
class StopExperiment(Event):
    exp_name: str


@dataclasses.dataclass
class ModelCreation(Event):
    model_name: str
    metadata: dict[str, Any]


@dataclasses.dataclass
class RecordMetadata(Event):
    model_name: str
    name: str
    metadata: dict[str, Any]


@dataclasses.dataclass
class SaveCheckpoint(Event):
    model_name: str
    definition: str
    location: str
    epoch: int


@dataclasses.dataclass
class LoadCheckpoint(Event):
    model_name: str
    definition: str
    location: str
    epoch: int


@dataclasses.dataclass
class StartTraining(Event):
    model_name: str
    start_epoch: int
    end_epoch: int


@dataclasses.dataclass
class StartEpoch(Event):
    epoch: int
    final_epoch: Optional[int] = None


@dataclasses.dataclass
class EndEpoch(Event):
    pass


@dataclasses.dataclass
class EpochBar(Event):
    source: str
    batch_size: int
    dataset_size: int
    push_updates: list[
        Callable[[Mapping[str, Any]], None]
    ] = dataclasses.field(default_factory=list)

    def update(self, metrics: Mapping[str, Any]) -> None:
        for update in self.push_updates:
            update(metrics)
        return


@dataclasses.dataclass
class ModelDidNotConverge(Event):
    exception: BaseException


@dataclasses.dataclass
class TerminatedTraining(Event):
    epoch: int
    cause: Optional[str] = None


@dataclasses.dataclass
class EndTraining(Event):
    pass


@dataclasses.dataclass
class Test(Event):
    model_name: str
    test_name: str


@dataclasses.dataclass
class Metrics(Event):
    model_name: str
    source: str
    epoch: int
    metrics: Mapping[str, float]

from __future__ import annotations
import abc
import dataclasses
from collections.abc import Callable
from typing import Any, Optional

from src.dry_torch import protocols as p


class Event(metaclass=abc.ABCMeta):
    auto_publish: Callable[[Event], None] = lambda self: None

    def __post_init__(self):
        self.__class__.auto_publish(self)

    @classmethod
    def set_auto_publish(cls, func: Callable[[Event], None]) -> None:
        cls.auto_publish = func
        return


@dataclasses.dataclass
class StartExperiment(Event):
    exp_name: str


@dataclasses.dataclass
class StopExperiment(Event):
    exp_name: str


@dataclasses.dataclass
class ModelCreation(Event):
    model: p.ModelProtocol
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RecordMetadata(Event):
    model_name: str
    class_name: str
    name: str
    kwargs: dict[str, Any]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SaveCheckpoint(Event):
    definition: str
    location: str


@dataclasses.dataclass
class LoadCheckpoint(Event):
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
    loader: p.LoaderProtocol
    push_updates: list[
        Callable[[dict[str, float]], None]
    ] = dataclasses.field(default_factory=list)

    def update_pbar(self, metrics: dict[str, float]) -> None:
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
class StartTest(Event):
    model_name: str
    test_name: str


@dataclasses.dataclass
class MetricsCreation(Event):
    model_name: str
    source: str
    epoch: int
    metrics: dict[str, float]

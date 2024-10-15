import abc
import dataclasses
import functools
from collections.abc import KeysView, Callable
from typing import Any, Optional, cast, Self

from src.dry_torch import descriptors
from src.dry_torch import protocols as p

class Event(metaclass=abc.ABCMeta):
    auto_publish: Callable[[Self], None] = lambda self: None

    def __post_init__(self):
        self.__class__.auto_publish(self)

@dataclasses.dataclass
class StartExperiment(Event):
    exp_name: str

@dataclasses.dataclass
class StopExperiment(Event):
    exp_name: str

@dataclasses.dataclass
class ModelCreation(Event):
    model: p.ModelProtocol

@dataclasses.dataclass
class CreateEvaluation(Event):
    model: p.ModelProtocol
    cls_str: str
    args: list[Any]
    kwargs: dict[str, Any]


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
    start_epoch: int = 1
    end_epoch: int = 2


@dataclasses.dataclass
class StartEpoch:
    epoch: int

@dataclasses.dataclass
class EpochProgressBar(Event):
    loader: p.LoaderProtocol
    push_updates: list[
        Callable[[str, float], None]
    ] = dataclasses.field(default_factory=list)


    def update(self, metric_name: str, metric_value: float) -> None:
        for update in self.push_updates:
            update(metric_name, metric_value)
        return


@dataclasses.dataclass
class TerminatedTraining(Event):
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
    partition: descriptors.Split
    epoch: int
    metrics: dict[str, float]


class Subscriber(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self._exp_name: Optional[str] = None

    @property
    def exp_name(self) -> str:
        if self._exp_name is None:
            raise RuntimeError("Subscriber has not been activated.")
        return self._exp_name

    @exp_name.setter
    def exp_name(self, exp_name: str) -> None:
        self._exp_name = exp_name

    @functools.singledispatchmethod
    def notify(self, event: Event) -> None:
        return

    @classmethod
    def defined_events(cls) -> KeysView[type[Event]]:
        # noinspection PyUnresolvedReferences
        register = cast(functools.singledispatchmethod,
                        cls.notify.register.__self__) # type: ignore
        return  register.dispatcher.registry.keys()


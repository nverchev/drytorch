import abc
import dataclasses
import functools


@dataclasses.dataclass
class Event(metaclass=abc.ABCMeta):
    pass

@dataclasses.dataclass
class ExperimentStart(Event):
    exp_name: str

@dataclasses.dataclass
class ExperimentEnd(Event):
    exp_name: str

@dataclasses.dataclass
class ModelCreation(Event):
    model_name: str
    model_class: type

@dataclasses.dataclass
class SaveCheckpoint(Event):
    definition: str
    location: str
    success: bool = dataclasses.field(init=False)


@dataclasses.dataclass
class LoadCheckpoint(Event):
    definition: str
    location: str
    success: bool = dataclasses.field(init=False)

@dataclasses.dataclass
class Evaluation(Event):
    model_name: str
    test_name: str
    pass

@dataclasses.dataclass
class TrainingStart(Event):
    model_name: str
    start_epoch: int = 1
    end_epoch: int = 2

@dataclasses.dataclass
class TrainingStop(Event):
    pass

@dataclasses.dataclass
class TrainingEnd(Event):
    pass

@dataclasses.dataclass
class MetricsCreation(Event):
    model_name: str
    source: str
    partition: str
    epoch: int
    metrics: dict[str, float]


class Subscriber(metaclass=abc.ABCMeta):

    @functools.singledispatchmethod
    @abc.abstractmethod
    def log(self, event: Event) -> None:
        return


class BuiltinLogger(Subscriber):

    @functools.singledispatchmethod
    def log(self, event: Event) -> None:
        return

    @log.register
    def _(self, event: TrainingEnd) -> None:
        print(event)
        return

BuiltinLogger().log(TrainingEnd())
BuiltinLogger().log(TrainingStart('test'))

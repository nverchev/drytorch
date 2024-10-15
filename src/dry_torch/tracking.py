from __future__ import annotations

import pathlib
import datetime
from typing import Any, Optional, Final, TypeVar, Generic

from src.dry_torch import repr_utils
from src.dry_torch import events
from src.dry_torch import exceptions
from src.dry_torch import builtin_logger

from src.dry_torch.backends import tqdm_backend

DEFAULT_SUBSCRIBER_CLASSES: list[type[events.Subscriber]] = [
    builtin_logger.InfoLogger,
    tqdm_backend.Tqdm
]

_T = TypeVar('_T')


class Experiment(Generic[_T]):
    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    _current_config: Optional[_T] = None
    _default_link_name = repr_utils.DefaultName('outputs')

    """
    This class is used to describe the experiment.

    Args:
        name: the model_name of the experiment.
        config: configuration for the experiment.
        pardir: parent directory for the folders with the module checkpoints
        save_metadata: whether to extract metadata from classes that 
        implement the save_metadata decorator
        max_items_repr: limits the size of iterators and arrays.



    Attributes:
        metric_logger: contains the saved metric_name and a plotting function
        epoch: the current epoch, that is, the number of epochs the module has 
        been trainer plus one.


    """

    def __init__(self,
                 name: str = '',
                 pardir: str | pathlib.Path = pathlib.Path(''),
                 config: Optional[Any] = None) -> None:

        self.name: Final = name or datetime.datetime.now().isoformat()
        self.pardir = pathlib.Path(pardir)
        self.dir = self.pardir / name
        self.dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        self.__class__.past_experiments.add(self)
        self.subscribers: dict[type[events.Event], list[events.Subscriber]] = {}
        for subscriber_class in DEFAULT_SUBSCRIBER_CLASSES:
            self.register_subscriber(subscriber_class())
        self.activate()

    def publish(self, event: events.Event) -> None:
        for subscriber in self.subscribers.get(event.__class__, []):
            subscriber.notify(event)

    def register_subscriber(self, subscriber: events.Subscriber) -> None:
        subscriber.exp_name = self.name
        for event in subscriber.defined_events():
            self.subscribers.setdefault(event, []).append(subscriber)
        return

    def activate(self) -> None:
        if Experiment._current is not None:
            self.stop()
        events.Event.auto_publish = self.publish
        # session = Session()
        Experiment._current = self
        self.__class__._current_config = self.config
        events.StartExperiment(self.name)
        return

    def stop(self) -> None:
        """"""
        Experiment._current = None
        event = events.StopExperiment(self.name)
        self.publish(event)
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(name={self.name})'

    @classmethod
    def current(cls) -> Experiment:
        if Experiment._current is not None:
            return Experiment._current
        unnamed_experiment = cls(datetime.datetime.now().isoformat())
        unnamed_experiment.activate()
        return unnamed_experiment

    @classmethod
    def get_config(cls) -> _T:
        cfg = cls._current_config
        if cfg is None:
            raise exceptions.NoConfigError()
        return cfg



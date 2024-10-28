from __future__ import annotations

import abc
import functools
import pathlib
import datetime
import weakref
from abc import abstractmethod
from typing import Optional, Final, TypeVar, Generic, KeysView, cast
import warnings

from src.dry_torch import repr_utils
from src.dry_torch import events
from src.dry_torch import exceptions

_T = TypeVar('_T')


class Tracker(metaclass=abc.ABCMeta):
    priority = 1

    def __init__(self) -> None:
        weakref.finalize(self, self.notify, events.StopExperiment(''))

    @functools.singledispatchmethod
    @abstractmethod
    def notify(self, event: events.Event) -> None:
        return

    @classmethod
    def defined_events(cls) -> KeysView[type[events.Event]]:
        # noinspection PyUnresolvedReferences
        register = cast(functools.singledispatchmethod,
                        cls.notify.register.__self__)  # type: ignore
        return register.dispatcher.registry.keys()

    def __gt__(self, other: Tracker) -> bool:
        return self.priority > other.priority


class Handler(Tracker, metaclass=abc.ABCMeta):
    priority = 2


class Logger(Tracker, metaclass=abc.ABCMeta):
    priority = 0


# Default specified in __init__.py
DEFAULT_TRACKERS: list[Tracker] = []


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
                 config: Optional[_T] = None) -> None:

        self.name: Final = name or datetime.datetime.now().isoformat()
        self.dir = pathlib.Path(pardir) / name
        self.dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        self.__class__.past_experiments.add(self)
        self.named_trackers: dict[str, Tracker] = {}
        self.event_trackers: dict[type[events.Event], list[Tracker]] = {}
        for tracker in DEFAULT_TRACKERS:
            self.register_tracker(tracker)

    def register_tracker(self, tracker: Tracker) -> None:
        tracker_name = tracker.__class__.__name__
        if tracker_name in self.named_trackers:
            raise exceptions.TrackerAlreadyRegisteredError(tracker_name,
                                                           self.name)
        self.named_trackers[tracker_name] = tracker

        for event_class in list(tracker.defined_events()):
            self.event_trackers.setdefault(event_class, []).append(tracker)
            self.event_trackers[event_class].sort(reverse=True)
        return

    def remove_named_tracker(self, tracker_name: str) -> None:
        try:
            tracker = self.named_trackers.pop(tracker_name)
        except ValueError:
            raise exceptions.TrackerNotRegisteredError(tracker_name, self.name)
        else:
            for event_class in tracker.defined_events():
                self.event_trackers[event_class].remove(tracker)
        return

    def remove_all_trackers(self) -> None:
        for tracker_name in list(self.named_trackers):
            self.remove_named_tracker(tracker_name)
        return

    def publish(self, event: events.Event) -> None:
        event_trackers = self.event_trackers.get(event.__class__, [])
        for subscriber in event_trackers:
            try:
                subscriber.notify(event)
            except BaseException as be:
                raise be
                name = subscriber.__class__.__name__
                warnings.warn(exceptions.TrackerError(name, be))

    def start(self) -> None:
        if Experiment._current is not None:
            self.stop()
        events.Event.set_auto_publish(self.publish)
        Experiment._current = self
        self.__class__._current_config = self.config
        events.StartExperiment(self.name)
        return

    def stop(self) -> None:
        """"""
        if Experiment._current is not None:
            Experiment._current = None
            events.StopExperiment(self.name)
        return

    @classmethod
    def current(cls) -> Experiment:
        if Experiment._current is None:
            new_default_experiment = cls()
            new_default_experiment.start()
            return new_default_experiment
        return Experiment._current

    @classmethod
    def quit(cls) -> None:
        if Experiment._current is not None:
            Experiment._current.stop()
            return
        return

    @classmethod
    def get_config(cls) -> _T:
        cfg = cls._current_config
        if cfg is None:
            raise exceptions.NoConfigError()
        return cfg

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(name={self.name})'

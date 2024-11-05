from __future__ import annotations

import abc
import functools
import pathlib
import datetime
import weakref
from abc import abstractmethod
from typing import Optional, Final, TypeVar, Generic, KeysView, cast, Iterable
import warnings

from src.dry_torch import repr_utils
from src.dry_torch import log_events
from src.dry_torch import exceptions

_T = TypeVar('_T')


class Tracker(metaclass=abc.ABCMeta):
    """Abstract base class for tracking events with priority ordering.
    """

    def __init__(self) -> None:
        weakref.finalize(self, self.notify, log_events.StopExperiment(''))

    @functools.singledispatchmethod
    @abstractmethod
    def notify(self, event: log_events.Event) -> None:
        """Notify the tracker of an event.

        Args:
            event (log_events.Event): The event to notify about.
        """
        return

    @classmethod
    def defined_events(cls) -> KeysView[type[log_events.Event]]:
        """Return the types of events this tracker is registered for."""
        register = cast(functools.singledispatchmethod,
                        cls.notify.register.__self__)  # type: ignore
        return register.dispatcher.registry.keys()


DEFAULT_TRACKERS: dict[str, Tracker] = {}


class Experiment(Generic[_T]):
    """Manages experiment details, configuration, and event tracking.

    Args:
        name: The name of the experiment.
        pardir: Parent directory for experiment data.
        config: Configuration for the experiment.

    Attributes:
        name: The experiment name, set at initialization.
        dir: The directory for storing experiment files.
        config: Configuration object for the experiment.
    """
    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    _current_config: Optional[_T] = None
    _default_link_name = repr_utils.DefaultName('outputs')

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
        self.event_trackers: dict[type[log_events.Event], list[Tracker]] = {}
        self.register_trackers(*DEFAULT_TRACKERS.values())

    def register_tracker(self, tracker: Tracker) -> None:
        """Register a tracker to the experiment.

        Args:
            tracker: The tracker to register.

        Raises:
            TrackerAlreadyRegisteredError: If the tracker is already registered.
        """
        tracker_name = tracker.__class__.__name__
        if tracker_name in self.named_trackers:
            raise exceptions.TrackerAlreadyRegisteredError(tracker_name,
                                                           self.name)
        self.named_trackers[tracker_name] = tracker

        for event_class in list(tracker.defined_events()):
            self.event_trackers.setdefault(event_class, []).append(tracker)
            self.event_trackers[event_class].sort(reverse=True)

    def register_trackers(self, *trackers: Tracker) -> None:
        """Register trackers from am iterable to the experiment.

        Args:
            trackers: Trackers to register.

        Raises:
            TrackerAlreadyRegisteredError: If a tracker is already registered.
        """
        for tracker in trackers:
            self.register_tracker(tracker)

    def remove_named_tracker(self, tracker_name: str) -> None:
        """Remove a tracker by name from the experiment.

        Args:
            tracker_name: Name of the tracker to remove.

        Raises:
            TrackerNotRegisteredError: If the tracker is not registered.
        """
        try:
            tracker = self.named_trackers.pop(tracker_name)
        except KeyError:
            raise exceptions.TrackerNotRegisteredError(tracker_name, self.name)
        else:
            for event_class in tracker.defined_events():
                self.event_trackers[event_class].remove(tracker)

    def remove_all_trackers(self) -> None:
        """Remove all trackers from the experiment."""
        for tracker_name in list(self.named_trackers):
            self.remove_named_tracker(tracker_name)

    def publish(self, event: log_events.Event) -> None:
        """Publish an event to all registered trackers.

        Args:
            event: The event to publish.
        """
        event_trackers = self.event_trackers.get(event.__class__, [])
        for subscriber in event_trackers:
            try:
                subscriber.notify(event)
            except Exception as exc:
                name = subscriber.__class__.__name__
                warnings.warn(exceptions.TrackerError(name, exc))

    def start(self) -> None:
        """Start the experiment, setting it as the current active experiment."""
        if Experiment._current is not None:
            self.stop()
        log_events.Event.set_auto_publish(self.publish)
        Experiment._current = self
        self.__class__._current_config = self.config
        log_events.StartExperiment(self.name)

    def stop(self) -> None:
        """Stop the experiment, clearing it from the active experiment."""
        if Experiment._current is not None:
            Experiment._current = None
            log_events.StopExperiment(self.name)

    @classmethod
    def current(cls) -> Experiment:
        """Return the current active experiment if exists or start a new one.

        Returns:
            Experiment: The current active experiment.
        """
        if Experiment._current is None:
            new_default_experiment = cls()
            new_default_experiment.start()
            return new_default_experiment
        return Experiment._current

    @classmethod
    def quit(cls) -> None:
        """Stop the current experiment if one is active."""
        if Experiment._current is not None:
            Experiment._current.stop()

    @classmethod
    def get_config(cls) -> _T:
        """Retrieve the configuration of the current experiment.

        Returns:
            _T: Configuration object of the current experiment.

        Raises:
            NoConfigError: If there is no configuration available.
        """
        cfg = cls._current_config
        if cfg is None:
            raise exceptions.NoConfigError()
        return cfg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

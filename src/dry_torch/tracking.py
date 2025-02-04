"""
Module for coordinating logging of metadata, internal messages and metrics.

Attributes:
    DEFAULT_TRACKERS: named trackers that are automatically registered to the
                      experiment.
"""

from __future__ import annotations

import abc
from abc import abstractmethod
import functools
import pathlib
from types import TracebackType
from typing import Any, Generic, Optional, TypeVar
import warnings
import weakref

from dry_torch import exceptions
from dry_torch import log_events
from dry_torch import protocols as p
from dry_torch import repr_utils

_T = TypeVar('_T')

DEFAULT_TRACKERS: dict[str, Tracker] = {}


class MetadataManager:
    """
    Class that handles and generates metadata.

    Attributes:
        used_names: Set to keep track of already registered names.
        max_items_repr: Maximum number of documented items in an object.
    """

    def __init__(self, max_items_repr: int = 10) -> None:
        """
        Args:
            max_items_repr: Maximum number of documented items in an object.
        """
        super().__init__()
        self.used_names = set[str]()
        self.max_items_repr = max_items_repr

    def record_model_call(self, name: str, model_name: str, obj: Any) -> None:
        """
        Records metadata of a given named object.

        Args:
            name: The name associated to the object.
            model_name: The name of the model linked to the object.
            obj: The object to document.
        """
        self._register_name(name)
        metadata = self.extract_metadata(obj, max_size=self.max_items_repr)
        log_events.CallModel(name, model_name, metadata)
        return

    def register_model(self, model: p.ModelProtocol) -> None:
        """
        Records metadata of a given model.

        Args:
            model: The model to document.
        """
        self._register_name(model.name)
        metadata = {'module': repr_utils.LiteralStr(repr(model.module))}
        log_events.ModelCreation(model.name, metadata)
        return

    def _register_name(self, name: str) -> None:
        if name in self.used_names:
            raise exceptions.NameAlreadyRegisteredError(name)
        self.used_names.add(name)
        return

    @staticmethod
    def extract_metadata(obj: Any, max_size: int) -> dict[str, Any]:
        """
        Wrapper of recursive_repr that catches RecursionError.

        Args:
            obj: an object to document.
            max_size: maximum number of documented items in an obj.
        """
        try:
            metadata = repr_utils.recursive_repr(obj, max_size=max_size)
        except RecursionError:
            warnings.warn(exceptions.RecursionWarning())
            metadata = {}
        return metadata


class Tracker(metaclass=abc.ABCMeta):
    """Abstract base class for tracking events with priority ordering."""

    def __init__(self) -> None:
        weakref.finalize(self, self.notify, log_events.StopExperiment(''))

    @functools.singledispatchmethod
    @abstractmethod
    def notify(self, event: log_events.Event) -> None:
        """
        Notify the tracker of an event.

        Args:
            event: The event to notify about.
        """
        return


class EventDispatcher:
    """
    Notifies tracker of an event.

    Attributes:
        exp_name: Name of the current experiment.
        named_trackers: A dictionary of trackers, indexed by their names.
    """

    def __init__(self, exp_name) -> None:
        """
        Args:
            exp_name: Name of the current experiment.
        """
        self.exp_name = str(exp_name)
        self.named_trackers: dict[str, Tracker] = {}

    def publish(self, event: log_events.Event) -> None:
        """
        Publish an event to all registered trackers.

        Args:
            event: The event to publish.
        """
        for tracker in self.named_trackers.values():
            try:
                tracker.notify(event)
            except Exception as err:
                name = tracker.__class__.__name__
                warnings.warn(exceptions.TrackerError(name, err))

    def _register_tracker(self, name: str, tracker: Tracker) -> None:
        """
        Register a tracker to the experiment.

        Args:
            name: The name associated to the tracker.
            tracker: The tracker to register.

        Raises:
            TrackerAlreadyRegisteredError: If the tracker is already registered.
        """
        if name in self.named_trackers:
            raise exceptions.TrackerAlreadyRegisteredError(name, self.exp_name)

        self.named_trackers[name] = tracker

    def register(self, *trackers: Tracker, **named_trackers: Tracker) -> None:
        """
        Register trackers from am iterable to the experiment.

        Args:
            trackers: Trackers to register.

        Raises:
            TrackerAlreadyRegisteredError: If a tracker is already registered.
        """
        for tracker in trackers:
            name = tracker.__class__.__name__
            self._register_tracker(name, tracker)

        for name, tracker in named_trackers.items():
            self._register_tracker(name, tracker)

    def remove(self, tracker_name: str) -> None:
        """
        Remove a tracker by name from the experiment.

        Args:
            tracker_name: Name of the tracker to remove.

        Raises:
            TrackerNotRegisteredError: If the tracker is not registered.
        """
        try:
            self.named_trackers.pop(tracker_name)
        except KeyError:
            raise exceptions.TrackerNotRegisteredError(tracker_name,
                                                       self.exp_name)
        return

    def remove_all(self) -> None:
        """Remove all trackers from the experiment."""
        for tracker_name in list(self.named_trackers):
            self.remove(tracker_name)


class Experiment(Generic[_T]):
    """Manages experiment metadata, configuration, and tracking.

    Attributes:
        dir: The directory for storing experiment files.
        config: Configuration object for the experiment.
        metadata_manager: Manager for recording metadata.
        trackers: Dispatcher for publishing events.
    """

    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    _current_config: Optional[_T] = None
    _name = repr_utils.DefaultName()

    def __init__(self,
                 name: str = '',
                 par_dir: str | pathlib.Path = pathlib.Path(''),
                 config: Optional[_T] = None) -> None:
        """
        Args:
            name: The name of the experiment. Defaults to class name.
            par_dir: Parent directory for experiment data.
            config: Configuration for the experiment.
        """
        self._name = name
        self.dir = pathlib.Path(par_dir) / str(self.name)
        self.config = config
        self.metadata_manager = MetadataManager()
        self.trackers = EventDispatcher(self.name)
        self.trackers.register(**DEFAULT_TRACKERS)
        self.__class__.past_experiments.add(self)

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return self._name

    def __enter__(self) -> None:
        return self.start()

    def __exit__(self,
                 exc_type: type[BaseException],
                 exc_val: BaseException,
                 exc_tb: TracebackType) -> None:
        return self.stop()

    def start(self) -> None:
        """Start the experiment, setting it as the current active experiment."""
        if Experiment._current is not None:
            self.stop()
        log_events.Event.set_auto_publish(self.trackers.publish)
        Experiment._current = self
        self.__class__._current_config = self.config
        log_events.StartExperiment(self.name, self.dir)

    @staticmethod
    def stop() -> None:
        """Stop the experiment, clearing it from the active experiment."""
        if Experiment._current is not None:
            name = Experiment._current.name
            Experiment._current = None
            log_events.StopExperiment(name)

    @classmethod
    def current(cls) -> Experiment:
        """Return the current active experiment if exists or start a new one.

        Returns:
            Experiment: The current active experiment.
        Raises:
            NoActiveExperimentError: If there is no active experiment.
        """
        if Experiment._current is None:
            raise exceptions.NoActiveExperimentError()
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
            NoActiveExperimentError: If there is no active experiment.
            NoConfigError: If there is no configuration available.
        """
        cfg = cls._current_config
        if Experiment._current is None:
            raise exceptions.NoActiveExperimentError()
        if cfg is None:
            raise exceptions.NoConfigError()
        return cfg

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

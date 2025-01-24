from __future__ import annotations

import abc
import functools
import pathlib
import datetime
import weakref
from abc import abstractmethod
from typing import Optional, Final, TypeVar, Generic, Any
import warnings

from src.dry_torch import repr_utils
from src.dry_torch import log_events
from src.dry_torch import exceptions
from src.dry_torch import protocols as p

_T = TypeVar('_T')


class MetadataManager:
    def __init__(self, max_items_repr: int = 10) -> None:
        super().__init__()
        self.max_items_repr = max_items_repr

    def record_metadata(self, name: str, object: Any) -> None:
        metadata = self.extract_metadata(object, max_size=self.max_items_repr)
        log_events.RecordMetadata(name, metadata)
        return

    def register_model(self, model: p.ModelProtocol) -> None:
        name = model.name
        # self._model_names.setdefault(name, repr_utils.DefaultName(name))
        # model.name = self._model_names[name]()
        metadata = {'module': repr_utils.LiteralStr(repr(model.module))}
        log_events.ModelCreation(model.name, metadata)
        return

    # def record_metadata(self, model_name, class_name, name, kwargs) -> None:
    #     model_default_names = self.default_names.get(event.model_name)
    #     if model_default_names is None:
    #         raise exceptions.ModelNotExistingError(event.model_name,
    #                                                self.exp_name)
    #     cls_count = model_default_names.setdefault(
    #         event.name,
    #         repr_utils.DefaultName(event.name)
    #     )
    #     name = cls_count()
    #
    #     metadata = self.extract_metadata(event.kwargs, self.max_items_repr)
    #     event.kwargs['name'] = name
    #     metadata |= {event.class_name: metadata}
    #     record = log_events.RecordMetadata(model_name, name, metadata)
    #
    #     return

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
        """Notify the tracker of an event.

        Args:
            event (log_events.Event): The event to notify about.
        """
        return


DEFAULT_TRACKERS: dict[str, Tracker] = {}


class Experiment(Generic[_T]):
    """Manages experiment details, configuration, and event tracking.

    Args:
        name: The name of the experiment.
        par_dir: Parent directory for experiment data.
        config: Configuration for the experiment.

    Attributes:
        name: The experiment name, set at initialization.
        dir: The directory for storing experiment files.
        config: Configuration object for the experiment.
    """
    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    _current_config: Optional[_T] = None

    def __init__(self,
                 name: str = '',
                 par_dir: str | pathlib.Path = pathlib.Path(''),
                 config: Optional[_T] = None) -> None:
        self.name: Final = name or datetime.datetime.now().isoformat()
        self.dir = pathlib.Path(par_dir) / name
        self.dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        self.metadata_manager = MetadataManager()
        self.__class__.past_experiments.add(self)
        self.named_trackers: dict[str, Tracker] = {}
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
            self.named_trackers.pop(tracker_name)
        except KeyError:
            raise exceptions.TrackerNotRegisteredError(tracker_name, self.name)
        return

    def remove_all_trackers(self) -> None:
        """Remove all trackers from the experiment."""
        for tracker_name in list(self.named_trackers):
            self.remove_named_tracker(tracker_name)

    def publish(self, event: log_events.Event) -> None:
        """Publish an event to all registered trackers.

        Args:
            event: The event to publish.
        """
        for tracker in self.named_trackers.values():
            try:
                tracker.notify(event)
            except Exception as exc:
                name = tracker.__class__.__name__
                warnings.warn(exceptions.TrackerError(name, exc))

    def start(self) -> None:
        """Start the experiment, setting it as the current active experiment."""
        if Experiment._current is not None:
            self.stop()
        log_events.Event.set_auto_publish(self.publish)
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
            NoConfigError: If there is no configuration available.
        """
        cfg = cls._current_config
        if cfg is None:
            raise exceptions.NoConfigError()
        return cfg

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

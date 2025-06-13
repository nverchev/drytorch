"""
Module for coordinating logging of metadata, internal messages and metrics.

Attributes:
    DEFAULT_TRACKERS: named trackers registered to experiments by default.
"""

from __future__ import annotations

import abc
from abc import abstractmethod
import functools
from typing import Any, Self, cast
import warnings

from drytorch import exceptions
from drytorch import log_events
from drytorch import protocols as p
from drytorch.utils import repr_utils

DEFAULT_TRACKERS: dict[str, Tracker] = {}


class MetadataManager:
    """
    Class that handles and generates metadata.

    Attributes:
        used_names: set to keep track of already registered names.
        max_items_repr: maximum number of documented items in an object.
    """

    def __init__(self, max_items_repr: int = 10) -> None:
        """
        Args:
            max_items_repr: maximum number of documented items in an object.
        """
        super().__init__()
        self.used_names = set[str]()
        self.max_items_repr = max_items_repr

    def record_model_call(self, source: Any, model: p.ModelProtocol) -> None:
        """
        Record metadata of an object that calls the model.

        Args:
            source: the object calling the model.
            model: the model that is called.
        """

        source_name = getattr(source, 'name', '') or source.__class__.__name__
        source_version = self._get_version(source)
        model_version = self._get_version(model)
        self._register_name(source_name)
        metadata = self.extract_metadata(source, max_size=self.max_items_repr)
        log_events.CallModel(source_name,
                             source_version,
                             model.name,
                             model_version,
                             metadata)
        return

    def register_model(self, model: p.ModelProtocol) -> None:
        """
        Record metadata of a given model.

        Args:
            model: the model to document.
        """
        self._register_name(model.name)
        metadata = {'module': repr_utils.LiteralStr(repr(model.module))}
        model_version = self._get_version(model)
        log_events.ModelCreation(model.name, model_version, metadata)
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

    @staticmethod
    def _get_version(x: Any) -> str:
        return getattr(x, 'created_at', '') or repr_utils.Versioned().created_at


class Tracker(metaclass=abc.ABCMeta):
    """Abstract base class for tracking events with priority ordering."""

    @functools.singledispatchmethod
    @abstractmethod
    def notify(self, event: log_events.Event) -> None:
        """
        Notify the tracker of an event.

        Args:
            event: the event to notify about.
        """
        return

    def clean_up(self) -> None:
        """Override to clean up the tracker."""
        return

    @classmethod
    def current(cls) -> Self:
        """
        Get the registered tracker that is already registered.

        Returns:
            The instance of the tracker registered to the current experiment.

        Raises:
            TrackerNotRegisteredError: if the tracker is not registered.
        """
        from drytorch.experiments import Experiment
        exp = Experiment.current()
        try:
            self = exp.trackers.named_trackers[cls.__name__]
        except KeyError:
            raise exceptions.TrackerNotRegisteredError(cls.__name__, exp.name)

        return cast(Self, self)


class EventDispatcher:
    """
    Notifies tracker of an event.

    Attributes:
        exp_name: name of the current experiment.
        named_trackers: a dictionary of trackers, indexed by their names.
    """

    def __init__(self, exp_name) -> None:
        """
        Args:
            exp_name: name of the current experiment.
        """
        self.exp_name = str(exp_name)
        self.named_trackers: dict[str, Tracker] = {}
        return

    def publish(self, event: log_events.Event) -> None:
        """
        Publish an event to all registered trackers.

        Args:
            event: the event to publish.
        """
        to_be_removed = list[str]()
        for tracker in self.named_trackers.values():
            try:
                tracker.notify(event)
            except Exception as err:
                name = tracker.__class__.__name__
                warnings.warn(exceptions.TrackerError(name, err))
                tracker.clean_up()
                to_be_removed.append(name)

        for name in to_be_removed:
            self.remove(name)

        return

    def _register_tracker(self, name: str, tracker: Tracker) -> None:
        """
        Register a tracker to the experiment.

        Args:
            name: the name associated with the tracker.
            tracker: the tracker to register.

        Raises:
            TrackerAlreadyRegisteredError: if the tracker is already registered.
        """
        if name in self.named_trackers:
            raise exceptions.TrackerAlreadyRegisteredError(name, self.exp_name)

        self.named_trackers[name] = tracker
        return

    def register(self, *trackers: Tracker, **named_trackers: Tracker) -> None:
        """
        Register trackers from am iterable to the experiment.

        Args:
            trackers: trackers to register.

        Raises:
            TrackerAlreadyRegisteredError: if a tracker is already registered.
        """
        for tracker in trackers:
            name = tracker.__class__.__name__
            self._register_tracker(name, tracker)

        for name, tracker in named_trackers.items():
            self._register_tracker(name, tracker)

        return

    def remove(self, tracker_name: str) -> None:
        """
        Remove a tracker by name from the experiment.

        Args:
            tracker_name: name of the tracker to remove.

        Raises:
            TrackerNotRegisteredError: if the tracker is not registered.
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

        return


def extend_default_trackers(tracker_list: list[Tracker]) -> None:
    """Add a list of trackers to the default ones."""
    for tracker in tracker_list:
        DEFAULT_TRACKERS[tracker.__class__.__name__] = tracker

    return


def remove_all_default_trackers() -> None:
    """Remove all default trackers."""
    DEFAULT_TRACKERS.clear()
    return

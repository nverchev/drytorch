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
from typing import Any
import warnings
import weakref

from dry_torch import exceptions
from dry_torch import log_events
from dry_torch import protocols as p
from dry_torch import repr_utils

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

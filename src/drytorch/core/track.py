"""Module for coordinating logging of metadata, internal messages and metrics.

Attributes:
    DEFAULT_TRACKERS: named trackers registered to experiments by default.
"""

from __future__ import annotations

import abc
import datetime
import functools
import warnings

from abc import abstractmethod
from typing import Any, Self

from drytorch.core import exceptions, log_events
from drytorch.core import protocols as p
from drytorch.utils import repr_utils


DEFAULT_TRACKERS: dict[str, Tracker] = {}


class MetadataManager:
    """Class that handles and generates metadata.

    Attributes:
        used_names: set to keep track of already registered names.
    """

    def __init__(
        self, max_depth_repr: int = 10, max_items_repr: int = 10
    ) -> None:
        """Constructor.

        Args:
            max_depth_repr: maximum number of recursion when documenting items.
            max_items_repr: maximum number of documented items in an object.
        """
        super().__init__()
        self.used_names = set[str]()

    def register_actor(
        self, actor: Any, model: p.ModelProtocol[Any, Any]
    ) -> None:
        """Record metadata of an object that acts on a model.

        Args:
            actor: the object acting on the model.
            model: the model that is called.
        """
        actor_name = getattr(actor, 'name', '') or actor.__class__.__name__
        actor_version = self._get_ts(actor)
        model_version = self._get_ts(model)
        self._register_name(actor_name)
        metadata = self.extract_metadata(actor)
        log_events.ActorRegistrationEvent(
            actor_name, actor_version, model.name, model_version, metadata
        )
        return

    def register_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Record metadata of a given model.

        Args:
            model: the model to document.
        """
        self._register_name(model.name)
        metadata = {'module': repr_utils.LiteralStr(repr(model.module))}
        model_version = self._get_ts(model)
        log_events.ModelRegistrationEvent(model.name, model_version, metadata)
        return

    def unregister_actor(self, actor: Any) -> None:
        """Unregister an object that acts on a model.

        Args:
            actor: the object actin on the model.
        """
        caller_name = getattr(actor, 'name', '') or actor.__class__.__name__
        self._unregister_name(caller_name)
        return

    def unregister_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Record metadata of a given model.

        Args:
            model: the model to document.
        """
        self._unregister_name(model.name)
        return

    def _register_name(self, name: str) -> None:
        if name in self.used_names:
            raise exceptions.NameAlreadyRegisteredError(name)

        self.used_names.add(name)
        return

    def _unregister_name(self, name: str) -> None:
        self.used_names.remove(name)
        return

    @staticmethod
    def extract_metadata(obj: Any) -> dict[str, Any]:
        """Wrapper of recursive_repr that catches RecursionError.

        To change the maximum number of documented items in an obj,
        set a maximum number of recursive call, and include properties,
        see the global variables in utils.repr_utils.

        Args:
            obj: an object to document.
            depth: maximum number of recursion when documenting items.
        """
        try:
            metadata = repr_utils.recursive_repr(obj)
        except RecursionError:
            # noinspection PyArgumentEqualDefault
            warnings.warn(exceptions.RecursionWarning(), stacklevel=1)
            metadata = {}

        return metadata

    @staticmethod
    def _get_ts(x: Any) -> datetime.datetime:
        possible_version = getattr(x, 'created_at', None)
        if isinstance(possible_version, datetime.datetime):
            return possible_version
        else:
            return repr_utils.CreatedAtMixin().created_at


class Tracker(metaclass=abc.ABCMeta):
    """Abstract base class for tracking events with priority ordering."""

    _current: Self | None = None

    @functools.singledispatchmethod
    @abstractmethod
    def notify(self, event: log_events.Event) -> None:
        """Notify the tracker of an event.

        Args:
            event: the event to notify about.
        """
        return

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        _not_used = event
        self._set_current(self)
        return

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        _not_used = event
        self._reset_current()
        return

    def clean_up(self) -> None:
        """Override to clean up the tracker."""
        return

    @classmethod
    def get_current(cls) -> Self:
        """Get the registered tracker that is already registered.

        Returns:
            The instance of the tracker registered to the current experiment.

        Raises:
            TrackerNotRegisteredError: if the tracker is not registered.
        """
        if cls._current is None:
            raise exceptions.TrackerNotActiveError(cls.__name__)
        return cls._current

    @classmethod
    def _set_current(cls, tracker: Self) -> None:
        cls._current = tracker
        return

    @classmethod
    def _reset_current(cls) -> None:
        cls._current = None
        return


class EventDispatcher:
    """Notifies tracker of an event.

    Attributes:
        exp_name: name of the current experiment.
        named_trackers: a dictionary of trackers, indexed by their names.
    """

    def __init__(self, exp_name) -> None:
        """Constructor.

        Args:
            exp_name: name of the current experiment.
        """
        self.exp_name = str(exp_name)
        self.named_trackers: dict[str, Tracker] = {}
        return

    def publish(self, event: log_events.Event) -> None:
        """Publish an event to all registered trackers.

        Args:
            event: the event to publish.
        """
        to_be_removed = list[str]()
        for tracker in self.named_trackers.values():
            try:
                tracker.notify(event)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as err:  # pylint: disable=broad-except
                name = tracker.__class__.__name__
                # noinspection PyArgumentEqualDefault
                warnings.warn(
                    exceptions.TrackerExceptionWarning(name, err), stacklevel=1
                )
                tracker.clean_up()
                to_be_removed.append(name)

        for name in to_be_removed:
            self.remove(name)

        return

    def _register_tracker(self, name: str, tracker: Tracker) -> None:
        """Register a tracker to the experiment.

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
        """Register trackers from am iterable to the experiment.

        Args:
            trackers: trackers to register with their class names.
            named_trackers: tracker to register with custom names.

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
        """Remove a tracker by name from the experiment.

        Args:
            tracker_name: name of the tracker to remove.

        Raises:
            TrackerNotRegisteredError: if the tracker is not registered.
        """
        try:
            self.named_trackers.pop(tracker_name)
        except KeyError as ke:
            raise exceptions.TrackerNotActiveError(tracker_name) from ke
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

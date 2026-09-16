"""Coordination of metadata, message, and metric tracking.

Attributes:
    DEFAULT_TRACKERS: named trackers registered to experiments by default.
"""

from __future__ import annotations

import abc
import datetime
import functools
import warnings

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Final, Self

from drytorch.core import exceptions, log_events
from drytorch.core import protocols as p
from drytorch.utils import repr_utils


__all__ = [
    'DEFAULT_TRACKERS',
    'EventDispatcher',
    'MetadataManager',
    'Tracker',
    'extend_default_trackers',
    'remove_all_default_trackers',
]


DEFAULT_TRACKERS: dict[str, Tracker] = {}


class MetadataManager:
    """Class that handles and generates metadata."""

    _named_metadata: Final[dict[str, Any]]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self._named_metadata = {}
        return

    @property
    def named_metadata(self) -> Mapping[str, Any]:
        """Mapping of registered names to their metadata."""
        return self._named_metadata

    def register_actor(
        self, actor: Any, model: p.ModelProtocol[Any, Any]
    ) -> None:
        """Record metadata of an object that acts on a model.

        Args:
            actor: the object acting on the model.
            model: the model that is called.
        """
        if isinstance(actor, p.NamedProtocol) and actor.name:
            actor_name = actor.name
        else:
            actor_name = actor.__class__.__name__

        actor_version = self._get_ts(actor)
        model_version = self._get_ts(model)
        metadata = self.extract_metadata(actor)
        self._register_metadata(actor_name, metadata)
        log_events.ActorRegistrationEvent(
            actor_name, actor_version, model.name, model_version, metadata
        )
        return

    def register_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Record metadata of a given model.

        Args:
            model: the model to document.
        """
        metadata = repr_utils.LiteralStr(repr(model.module))
        self._register_metadata(model.name, metadata)
        model_version = self._get_ts(model)
        log_events.ModelRegistrationEvent(model.name, model_version, metadata)
        return

    def unregister_actor(self, actor: Any) -> None:
        """Unregister an object that acts on a model.

        Args:
            actor: the object actin on the model.
        """
        if isinstance(actor, p.NamedProtocol) and actor.name:
            caller_name = actor.name
        else:
            caller_name = actor.__class__.__name__

        self._unregister_metadata(caller_name)
        return

    def unregister_model(self, model: p.ModelProtocol[Any, Any]) -> None:
        """Record metadata of a given model.

        Args:
            model: the model to document.
        """
        self._unregister_metadata(model.name)
        return

    def _register_metadata(self, name: str, metadata: Any) -> None:
        if name in self._named_metadata:
            raise exceptions.NameAlreadyRegisteredError(name)

        self._named_metadata[name] = metadata
        return

    def _unregister_metadata(self, name: str) -> None:
        if name in self._named_metadata:
            del self._named_metadata[name]

        return

    @staticmethod
    def extract_metadata(obj: Any) -> Mapping[str, Any] | str:
        """Wrapper of recursive_repr that catches RecursionError.

        To change the maximum number of documented items in an obj,
        set a maximum number of recursive calls and include properties,
        see the global variables in utils.repr_utils.

        Args:
            obj: an object to document.
        """
        try:
            metadata: Mapping[str, Any] | str = repr_utils.recursive_repr(obj)
        except RecursionError:
            warnings.warn(exceptions.RecursionWarning(), stacklevel=1)
            metadata = {}

        return metadata

    @staticmethod
    def _get_ts(x: Any) -> datetime.datetime:
        if isinstance(x, p.CreatedAtProtocol):
            if isinstance(x.created_at, datetime.datetime):
                return x.created_at

        return repr_utils.CreatedAtMixin().created_at


class Tracker(abc.ABC):
    """Abstract base class for tracking events with priority ordering."""

    _current: ClassVar[Self | None] = None

    @functools.singledispatchmethod
    @abc.abstractmethod
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
    def _(self, event: log_events.PauseExperimentEvent) -> None:
        _not_used = event
        self._reset_current()
        return

    @notify.register
    def _(self, event: log_events.ContinueExperimentEvent) -> None:
        _not_used = event
        self._set_current(self)
        return

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        _not_used = event
        self._reset_current()
        self.clean_up()
        return

    def clean_up(self) -> None:
        """Override to clean up the tracker idempotently."""
        return

    def close(self) -> None:
        """Release everything, including any paused state.

        Called once when the process ends.
        """
        self.clean_up()
        return

    @classmethod
    def get_current(cls) -> Self:
        """Get the registered tracker that is already registered.

        Returns:
            The instance of the tracker registered to the current experiment.

        Raises:
            TrackerNotActiveError: if the tracker is not registered.
        """
        if cls._current is None:
            raise exceptions.TrackerNotUsedError(cls.__name__)

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
    """

    exp_name: Final[str]
    _named_trackers: Final[dict[str, Tracker]]

    def __init__(self, exp_name) -> None:
        """Initialize.

        Args:
            exp_name: name of the current experiment.
        """
        self.exp_name = str(exp_name)
        self._named_trackers = {}
        return

    @property
    def named_trackers(self) -> Mapping[str, Tracker]:
        """A mapping of trackers, indexed by their names."""
        return self._named_trackers

    def clean_up(self) -> None:
        """Clean up all registered trackers."""
        for name, tracker in self._named_trackers.items():
            self._clean_up_tracker(name, tracker)

        return

    def publish(self, event: log_events.Event) -> None:
        """Publish an event to all registered trackers.

        Args:
            event: the event to publish.

        Raises:
            KeyboardInterrupt: if a tracker raises KeyboardInterrupt.
            SystemExit: if a tracker raises SystemExit.
        """
        to_be_removed: list[str] = []
        for name, tracker in self._named_trackers.items():
            try:
                tracker.notify(event)
            except (KeyboardInterrupt, SystemExit) as se:
                raise se

            except Exception as err:
                warnings.warn(
                    exceptions.TrackerExceptionWarning(name, err), stacklevel=1
                )
                to_be_removed.append(name)

        for name in to_be_removed:
            self._clean_up_tracker(name, self._named_trackers[name])
            self.remove(name)

        return

    def remove(self, tracker_name: str) -> None:
        """Remove a tracker by name from the dispatcher.

        Args:
            tracker_name: name of the tracker to remove.

        Raises:
            TrackerNotActiveError: if the tracker is not registered.
        """
        try:
            self._named_trackers.pop(tracker_name)
        except KeyError as ke:
            raise exceptions.TrackerNotUsedError(tracker_name) from ke
        return

    def remove_all(self) -> None:
        """Remove all trackers from the dispatcher."""
        for tracker_name in list(self._named_trackers):
            self.remove(tracker_name)

        return

    def subscribe(self, *trackers: Tracker, **named_trackers: Tracker) -> None:
        """Subscribe trackers to the dispatcher.

        Args:
            trackers: trackers to register with their class names.
            named_trackers: trackers to register with custom names.

        Raises:
            TrackerAlreadyRegisteredError: if a tracker is already registered.
        """
        for tracker in trackers:
            name = tracker.__class__.__name__
            self._subscribe_tracker(name, tracker)

        for name, tracker in named_trackers.items():
            self._subscribe_tracker(name, tracker)

        return

    def _clean_up_tracker(self, name: str, tracker: Tracker) -> None:
        try:
            tracker._reset_current()
            tracker.close()
        except Exception as err:
            warnings.warn(
                exceptions.TrackerExceptionWarning(name, err),
                stacklevel=2,
            )
        return

    def _subscribe_tracker(self, name: str, tracker: Tracker) -> None:
        """Subscribe a tracker to the dispatcher.

        Args:
            name: the name associated with the tracker.
            tracker: the tracker to register.

        Raises:
            TrackerAlreadyRegisteredError: if the tracker is already registered.
        """
        if name in self._named_trackers:
            raise exceptions.TrackerAlreadyRegisteredError(name, self.exp_name)

        self._named_trackers[name] = tracker
        return


def extend_default_trackers(tracker_list: Iterable[Tracker]) -> None:
    """Add trackers to the default ones."""
    for tracker in tracker_list:
        DEFAULT_TRACKERS[tracker.__class__.__name__] = tracker

    return


def remove_all_default_trackers() -> None:
    """Remove all default trackers."""
    DEFAULT_TRACKERS.clear()
    return

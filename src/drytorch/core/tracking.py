"""Coordination of metadata, message, and metric tracking.

Attributes:
    DEFAULT_TRACKERS: named trackers registered to experiments by default.
"""

from __future__ import annotations

import abc
import datetime
import functools
import warnings
import weakref

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Final, Protocol, Self, runtime_checkable

import torch

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


@runtime_checkable
class _CreatedAtProtocol(Protocol):
    """Protocol for objects exposing a creation timestamp."""

    @property
    @abc.abstractmethod
    def created_at(self) -> datetime.datetime:
        """The creation timestamp of the object."""


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

    def register_runner(
        self, runner: Any, model: p.ModelProtocol[Any, Any]
    ) -> None:
        """Record metadata of an object that runs a model.

        Args:
            runner: the object running the model.
            model: the model that is called.
        """
        if isinstance(runner, p.NamedProtocol) and runner.name:
            runner_name = runner.name
        else:
            runner_name = runner.__class__.__name__

        runner_version = self._get_ts(runner)
        model_version = self._get_ts(model)
        metadata = self.extract_metadata(runner)
        self._register_metadata(runner_name, metadata)
        log_events.RunnerRegistrationEvent(
            runner_name, runner_version, model.name, model_version, metadata
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
        n_parameters, n_trainable = self._count_parameters(model.module)
        log_events.ModelRegistrationEvent(
            model.name,
            model_version,
            metadata,
            n_parameters,
            n_trainable,
        )
        return

    def unregister_runner(self, runner: Any) -> None:
        """Unregister an object that runs a model.

        Args:
            runner: the object running the model.
        """
        if isinstance(runner, p.NamedProtocol) and runner.name:
            caller_name = runner.name
        else:
            caller_name = runner.__class__.__name__

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
    def _count_parameters(module: torch.nn.Module, /) -> tuple[int, int]:
        n_parameters = 0
        n_trainable = 0
        for param in module.parameters():
            n_parameters += param.numel()
            if param.requires_grad:
                n_trainable += param.numel()

        return n_parameters, n_trainable

    @staticmethod
    def _get_ts(x: Any) -> datetime.datetime:
        if isinstance(x, _CreatedAtProtocol):
            if isinstance(x.created_at, datetime.datetime):
                return x.created_at

        return repr_utils.CreatedAtMixin().created_at


class Tracker(abc.ABC):
    """Abstract base class for tracking events with priority ordering."""

    _current: ClassVar[Tracker | None] = None

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
        """Override to release the tracker resources idempotently."""
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
        if not isinstance(cls._current, cls):
            raise exceptions.TrackerNotUsedError(cls.__name__)

        return cls._current

    @classmethod
    def _set_current(cls, tracker: Tracker) -> None:
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
    _finalizer: Final[weakref.finalize]
    _named_trackers: Final[dict[str, Tracker]]

    def __init__(self, exp_name) -> None:
        """Initialize.

        Args:
            exp_name: name of the current experiment.
        """
        self.exp_name = str(exp_name)
        self._named_trackers = {}
        self._finalizer = weakref.finalize(
            self, self._close_trackers, self._named_trackers
        )
        return

    @property
    def named_trackers(self) -> Mapping[str, Tracker]:
        """A mapping of trackers, indexed by their names."""
        return self._named_trackers

    def close_all(self) -> None:
        """Close all registered trackers."""
        self._finalizer.detach()
        self._close_trackers(self._named_trackers)
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
            self._close_tracker(name, self._named_trackers[name])
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

    @staticmethod
    def _close_tracker(name: str, tracker: Tracker) -> None:
        try:
            tracker._reset_current()
            tracker.close()
        except Exception as err:
            warnings.warn(
                exceptions.TrackerExceptionWarning(name, err), stacklevel=2
            )

        return

    @staticmethod
    def _close_trackers(named_trackers: dict[str, Tracker]) -> None:
        for name, tracker in named_trackers.items():
            EventDispatcher._close_tracker(name, tracker)

        named_trackers.clear()
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

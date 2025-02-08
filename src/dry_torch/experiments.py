"""Module containing the experiment classes."""

from __future__ import annotations

import pathlib
from types import TracebackType
from typing import Generic, Optional, TypeVar

from dry_torch import exceptions
from dry_torch import log_events
from dry_torch import repr_utils
from dry_torch import tracking

_T = TypeVar('_T', covariant=True)
_U = TypeVar('_U', covariant=True)


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
        self.metadata_manager = tracking.MetadataManager()
        self.trackers = tracking.EventDispatcher(self.name)
        self.trackers.register(**tracking.DEFAULT_TRACKERS)
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

    def stop(self) -> None:
        """Stop the experiment, clearing it from the active experiment."""
        name = Experiment.current().name
        Experiment._current = None
        self.__class__._current_config = None
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


class ChildExperiment(Experiment[_T]):
    """Manages experiment metadata, configuration, and tracking.

    Attributes:
        dir: The directory for storing experiment files.
        config: Configuration object for the experiment.
        metadata_manager: Manager for recording metadata.
        trackers: Dispatcher for publishing events.
    """

    _current_config: Optional[_T] = None

    def __init__(self,
                 name: str,
                 config: Optional[_T] = None) -> None:
        """
        Args:
            name: The name of the experiment. Defaults to class name.
            config: Configuration for the experiment.
        """
        super().__init__(name, '', config)


class ParentExperiment(Experiment[_T], Generic[_T, _U]):
    """Manages experiment metadata, configuration, and tracking.

    Attributes:
        dir: The directory for storing experiment files.
        config: Configuration object for the experiment.
        metadata_manager: Manager for recording metadata.
        trackers: Dispatcher for publishing events.
    """
    children = list[ChildExperiment[_U]]()

    def register_child(self, child: ChildExperiment[_U]):
        """Register children experiments."""
        child.metadata_manager = self.metadata_manager
        for tracker in self.trackers.named_trackers.values():
            try:
                child.trackers.register(tracker)
            except exceptions.TrackerAlreadyRegisteredError:
                pass
        child.dir = self.dir / child.name
        self.children.append(child)

    @classmethod
    def get_child_config(cls) -> _U:
        """Get the configuration of the child that is currently active."""
        for child in cls.children:
            try:
                return child.get_config()
            except exceptions.NoConfigError:
                continue
        raise exceptions.NoConfigError

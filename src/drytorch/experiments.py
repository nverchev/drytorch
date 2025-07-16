"""Module containing the experiment classes."""

from __future__ import annotations

import pathlib
from types import TracebackType
from typing import Any, Generic, Optional, Self, TypeGuard, TypeVar

from typing_extensions import override

from drytorch import exceptions
from drytorch import log_events
from drytorch import tracking
from drytorch.utils import repr_utils

_T = TypeVar('_T', covariant=True)
_U = TypeVar('_U', covariant=True)


class Experiment(repr_utils.Versioned, Generic[_T]):
    """
    Manage experiment metadata, configuration, and tracking.

    Attributes:
        dir: the directory for storing experiment files.
        config: configuration object for the experiment.
        metadata_manager: manager for recording metadata.
        trackers: dispatcher for publishing events.
    """

    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    _current_config: Optional[_T] = None
    _name = repr_utils.DefaultName()

    def __init__(self,
                 name: str = '',
                 par_dir: str | pathlib.Path = pathlib.Path(),
                 config: Optional[_T] = None) -> None:
        """Constructor.

        Args:
            name: the name of the experiment. Defaults to class name.
            par_dir: parent directory for experiment data.
            config: configuration for the experiment.
        """
        super().__init__()
        self._name = name
        self.dir = pathlib.Path(par_dir) / self.name
        self.config = config
        self.metadata_manager = tracking.MetadataManager()
        self.trackers = tracking.EventDispatcher(self.name)
        self.trackers.register(**tracking.DEFAULT_TRACKERS)
        self.__class__.past_experiments.add(self)

    def __enter__(self) -> Self:
        """Start the experiment scope."""
        current_exp = Experiment._current
        if current_exp is not None:
            raise exceptions.NestedScopeError(current_exp.name, self.name)
        Experiment._current = self
        self.__class__._current_config = self.config
        log_events.Event.set_auto_publish(self.trackers.publish)
        log_events.StartExperiment(self.name,
                                   self.created_at,
                                   self.dir,
                                   self.config)
        return self

    def __exit__(self,
                 exc_type: type[BaseException],
                 exc_val: BaseException,
                 exc_tb: TracebackType) -> None:
        """Conclude the experiment scope."""
        log_events.StopExperiment(self.name)
        log_events.Event.set_auto_publish(None)
        Experiment._current = None
        self.__class__._current_config = None
        return

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    @property
    def name(self) -> str:
        """The name of the experiment plus a possible counter."""
        return self._name

    @classmethod
    def current(cls) -> Experiment:
        """
        Return the current active experiment if exists or start a new one.

        Returns:
            Experiment: the currently active experiment.
        Raises:
            NoActiveExperimentError: if there is no active experiment.
        """
        if Experiment._current is None:
            raise exceptions.NoActiveExperimentError()

        return Experiment._current

    @classmethod
    def get_config(cls) -> _T:
        """
        Retrieve the configuration of the current experiment.

        Returns:
            _T: configuration object of the current experiment.

        Raises:
            NoActiveExperimentError: if the experiment is not active.
            NoConfigError: if there is no configuration available.
        """
        if not cls._check_if_active():
            raise exceptions.NoActiveExperimentError(cls)

        config = cls._current_config
        if config is None:
            raise exceptions.NoConfigurationError()

        return config

    @classmethod
    def _check_if_active(cls) -> bool:
        return isinstance(Experiment.current(), cls)


class MainExperiment(Experiment[_T], Generic[_T, _U]):
    """
    Experiment containing smaller units (see SubExperiment).

    The experiment manages trackers, metadata and shared configurations (_T)
    and calls the currently active registered sub-experiment for the
    more specific configurations (_U).

    Attributes:
        dir: the directory for storing experiment files.
        config: configuration object for the experiment.
        metadata_manager: manager for recording metadata.
        trackers: dispatcher for publishing events.
    """
    sub_experiments: set[SubExperiment[_U]] = set()

    def register_sub(self, sub_exp: SubExperiment[_U]):
        """
        Register sub-experiments.

        Args:
            sub_exp: The sub experiment to register.
        """
        sub_exp.metadata_manager = self.metadata_manager
        for tracker in self.trackers.named_trackers.values():
            try:
                sub_exp.trackers.register(tracker)
            except exceptions.TrackerAlreadyRegisteredError:
                pass

        self.sub_experiments.add(sub_exp)
        sub_exp.main_experiment = self
        sub_exp.dir = self.dir / sub_exp.name
        return

    @classmethod
    def get_sub_config(cls) -> _U:
        """Return the configuration of the child that is currently active."""
        exp = Experiment.current()
        if not cls._is_registered(exp):
            raise exceptions.ActiveExperimentNotRegistered(exp.__class__, cls)

        config = exp.get_config()
        if config is None:
            raise exceptions.NoConfigurationError()

        return config

    @classmethod
    @override
    def _check_if_active(cls) -> bool:
        current_exp = Experiment.current()
        if current_exp in cls.sub_experiments:
            return True
        return isinstance(current_exp, cls)

    @classmethod
    def _is_registered(cls, exp: Experiment) -> TypeGuard[SubExperiment[_U]]:
        return exp in cls.sub_experiments


class SubExperiment(Experiment[_U]):
    """
    Experimental unit part of a larger experiment (see MainExperiment).

    This class handles different specifications for the same configuration (_U)
    concurring in the same main experiment.

    Attributes:
        dir: the directory for storing experiment files.
        config: configuration object for the experiment.
        metadata_manager: manager for recording metadata.
        trackers: dispatcher for publishing events.
    """
    main_experiment: Optional[MainExperiment[Any, _U]] = None

    def __init__(self,
                 name: str,
                 config: Optional[_U] = None) -> None:
        """Constructor.

        Args:
            name: The name of the experiment. Defaults to class name.
            config: Configuration for the experiment.
        """
        super().__init__(name, '', config)

    @override
    def __enter__(self) -> Self:
        if self.main_experiment is None:
            raise exceptions.SubExperimentNotRegisteredError(self.__class__)

        main_cls = self.main_experiment.__class__
        main_cls._current_config = self.main_experiment.config
        return super().__enter__()

    @override
    def __exit__(self,
                 exc_type: type[BaseException],
                 exc_val: BaseException,
                 exc_tb: TracebackType) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        if self.main_experiment is None:
            raise exceptions.SubExperimentNotRegisteredError(self.__class__)

        self.main_experiment.__class__._current_config = None
        return

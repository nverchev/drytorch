"""Module containing the experiment classes."""

from __future__ import annotations

import pathlib
from types import TracebackType
from typing import Generic, Optional, Self, TypeVar

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
    def current(cls) -> Self:
        """
        Return the current active experiment if exists or start a new one.

        Returns:
            Experiment: the currently active experiment.
        Raises:
            NoActiveExperimentError: if there is no active experiment.
        """
        if Experiment._current is None:
            raise exceptions.NoActiveExperimentError()

        if not isinstance(Experiment._current, cls):
            raise exceptions.NoActiveExperimentError(experiment_class=cls)

        return Experiment._current

    @classmethod
    def get_config(cls) -> _T:
        """
        Retrieve the configuration of the current experiment.

        Returns:
            _T: configuration object of the current experiment.
        Raises:
            NoConfigError: if there is no configuration available.
        """
        config = cls.current().config
        if config is None:
            raise exceptions.NoConfigurationError()

        return config

    @classmethod
    def _check_if_active(cls) -> bool:
        return isinstance(Experiment.current(), cls)


class ExperimentWithSpecs(Experiment[_T], Generic[_T, _U]):
    """
    Experiment containing specifications for shared config options.

    Attributes:
        dir: the directory for storing experiment files.
        config: configuration object for the experiment.
        metadata_manager: manager for recording metadata.
        trackers: dispatcher for publishing events.
    """

    def __init__(self,
                 name: str = '',
                 par_dir: str | pathlib.Path = pathlib.Path(),
                 config: Optional[_T] = None,
                 specs: Optional[_U] = None) -> None:
        """Constructor.

        Args:
            name: the name of the experiment. Defaults to class name.
            par_dir: parent directory for experiment data.
            config: configuration for the experiment.
            specs: specifications for the experiment.
        """
        super().__init__(name, par_dir, config)
        self.specs = specs

    @classmethod
    @override
    def current(cls) -> Self:
        return super().current()  # rewrite for better annotations

    @classmethod
    def get_specs(cls) -> _U:
        """Return the configuration of the child that is currently active."""
        exp: ExperimentWithSpecs[_T, _U] = ExperimentWithSpecs.current()
        specs = exp.specs
        if specs is None:
            raise exceptions.NoConfigurationError()

        return specs

    @classmethod
    def specify_experiment(cls,
                           base_exp: Experiment[_T],
                           name: str = '',
                           specs: Optional[_U] = None) -> Self:
        """Add specifications to an existing experiment."""
        instance = cls(name, base_exp.dir, base_exp.config, specs)
        instance.metadata_manager = base_exp.metadata_manager
        instance.trackers = base_exp.trackers
        return instance

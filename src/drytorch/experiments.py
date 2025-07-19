"""Module containing the experiment classes."""

from __future__ import annotations

import abc
import pathlib

from types import TracebackType
from typing import Any, Generic, Self, TypeVar

from drytorch import exceptions, log_events, tracking
from drytorch.utils import repr_utils


_T_co = TypeVar('_T_co', covariant=True)
_U_co = TypeVar('_U_co', covariant=True)


class Experiment(repr_utils.Versioned, Generic[_T_co]):
    """Manage experiment metadata, configuration, and tracking.

    Attributes:
        dir: the directory for storing experiment files.
        config: configuration object for the experiment.
        metadata_manager: manager for recording metadata.
        trackers: dispatcher for publishing events.
    """

    _current: Experiment | None = None
    _name = repr_utils.DefaultName()

    def __init__(
        self,
        name: str = '',
        par_dir: str | pathlib.Path = pathlib.Path(),
        config: _T_co | None = None,
    ) -> None:
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

    def __enter__(self) -> Self:
        """Start the experiment scope."""
        current_exp = Experiment._current
        if current_exp is not None:
            raise exceptions.NestedScopeError(current_exp.name, self.name)
        Experiment._current = self
        log_events.Event.set_auto_publish(self.trackers.publish)
        log_events.StartExperiment(
            self.name, self.created_at, self.dir, self.config
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        """Conclude the experiment scope."""
        log_events.StopExperiment(self.name)
        log_events.Event.set_auto_publish(None)
        Experiment._current = None
        return

    def __repr__(self) -> str:
        """Represent instance."""
        return f'{self.__class__.__name__}(name={self.name})'

    @property
    def name(self) -> str:
        """The name of the experiment plus a possible counter."""
        return self._name

    @classmethod
    def current(cls) -> Self:
        """Return the current active experiment if exists or start a new one.

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
    def get_config(cls) -> _T_co:
        """Retrieve the configuration of the current experiment.

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


class Specs(Generic[_U_co], metaclass=abc.ABCMeta):
    """Generic mixin for adding specifications to experiments.

    This class is useful when dividing an experiment in smaller units
    that have different similar internal configurations (here called
    specifications). This mixin creates sub-experiments that specify
    which internal configuration to use, avoiding code duplication.
    """

    metadata_manager: tracking.MetadataManager
    trackers: tracking.EventDispatcher

    def __init__(
        self,
        *args: Any,
        specs: _U_co | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructor.

        Args:
            args: positional arguments to pass on.
            specs: specifications for the experiment.
            kwargs: keyword arguments to pass on.
        """
        super().__init__(*args, **kwargs)
        self.specs: _U_co | None = specs

    @classmethod
    @abc.abstractmethod
    def current(cls) -> Any:
        """Method of the Experiment class."""

    @classmethod
    def get_specs(cls) -> _U_co:
        """Get specs from the currently active experiment."""
        exp = cls.current()
        if exp.specs is None:
            raise exceptions.NoConfigurationError()
        return exp.specs

    @classmethod
    def from_experiment(
        cls,
        base_exp: Experiment[_T_co],
        specs: _U_co | None = None,
    ) -> Self:
        """Create an ExperimentWithSpecs from an existing experiment."""
        instance = cls(
            name=base_exp.name,
            par_dir=base_exp.dir.parent,
            config=base_exp.config,
        )
        instance.metadata_manager = base_exp.metadata_manager
        instance.trackers = base_exp.trackers
        instance.specs = specs  # here to make it independent of mro
        return instance

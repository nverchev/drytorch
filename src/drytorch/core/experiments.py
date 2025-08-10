"""Module containing the Experiment and Run class."""

from __future__ import annotations

import pathlib

from types import TracebackType
from typing import Any, Final, Generic, Self, TypeVar

from drytorch.core import exceptions, log_events, tracking
from drytorch.utils import repr_utils

_T_co = TypeVar('_T_co', covariant=True)


class Experiment(repr_utils.Versioned, Generic[_T_co]):
    """Manage experiment configuration, directory, and tracking.

    This class associates a configuration file, a name and a working directory
    to a machine learning experiment. It also contains the trackers responsible
    for tracking the metadata and metrics for the experiment. Finally, it
    allows global access to a configuration file with the correct type
    annotations.

    Class Attributes:
        runs: previous runs from this class.

    Attributes:
        par_dir: parent directory for experiment data.
        tags: descriptors for the experiment.
        trackers: dispatcher for publishing events.
    """

    runs: list[Run[Any]] = []
    _name = repr_utils.DefaultName()
    __current: Experiment[Any] | None = None

    def __init__(
            self,
            config: _T_co,
            *,
            name: str = '',
            par_dir: str | pathlib.Path = pathlib.Path(),
            tags: list[str] | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: the name of the experiment. Default to class name.
            par_dir: parent directory for experiment data.
            config: configuration for the experiment.
            tags: descriptors for the experiment (e.g., "lr=0.01").

        """
        super().__init__()
        self.__config: Final[_T_co] = config
        self._name = name
        self.par_dir = pathlib.Path(par_dir)
        self.tags = tags or []
        self.trackers = tracking.EventDispatcher(self.name)
        self.trackers.register(**tracking.DEFAULT_TRACKERS)
        self._run: Run[_T_co] | None = None
        self._validate_name(name)
        return

    def __repr__(self) -> str:
        """Represent instance."""
        return f'{self.__class__.__name__}(name={self.name})'

    @property
    def name(self) -> str:
        """The name of the experiment plus a possible counter."""
        return self._name

    @property
    def config(self) -> _T_co:
        """Experiment configuration."""
        return self.__config

    def create_run(
            self,
            *,
            run_id: str | None = None,
            resume_last: bool = False,
    ) -> Run[_T_co]:
        """Convenience constructor for the Run class using this experiment.

        Args:
            run_id: identifier of the run, else a timestamp is used.
            resume_last: if id is not set, resume the previous run.
        """
        run = Run(
            experiment=self,
            run_id=run_id,
            resume_last_run=resume_last,
        )
        return run

    @property
    def run(self) -> Run[_T_co]:
        """Get current run.

        Raises:
            NoActiveExperimentError: if the experiment is not active.
        """
        if self._run is None:
            raise exceptions.NoActiveExperimentError(self.name)
        return self._run

    @run.setter
    def run(self, current_run: Run) -> None:
        self._run = current_run
        self.runs.append(current_run)
        return

    @classmethod
    def get_config(cls) -> _T_co:
        """Retrieve the configuration of the current experiment.

        Returns:
            _T: configuration object of the current experiment.
        """
        return cls.get_current().__config

    @classmethod
    def get_current(cls) -> Self:
        """Return the current active experiment if exists or start a new one.

        Returns:
            The currently active experiment.

        Raises:
            NoActiveExperimentError: if there is no active experiment.
        """
        if Experiment.__current is None:
            raise exceptions.NoActiveExperimentError()

        if not isinstance(Experiment.__current, cls):
            raise exceptions.NoActiveExperimentError(experiment_class=cls)

        return Experiment.__current

    @staticmethod
    def set_current(experiment: Experiment) -> None:
        """Set an experiment as the currently active experiment.

        Args:
            experiment: the experiment to activate.

        Raises:
            NestedScopeError: if there is already an active experiment.
        """
        if (old_exp := Experiment.__current) is not None:
            raise exceptions.NestedScopeError(old_exp.name, experiment.name)
        Experiment.__current = experiment
        return

    @staticmethod
    def clear_current() -> None:
        """Clear the currently active experiment.

        Raises:
            NoActiveExperimentError: if there is no active experiment.
        """
        if Experiment.__current is None:
            raise exceptions.NoActiveExperimentError()
        Experiment.__current = None
        return

    @staticmethod
    def _validate_name(name: str) -> None:
        not_allowed_chars = set(r'\/:*?"<>|')
        if invalid_chars := set(name) & not_allowed_chars:
            msg = f'Name contains invalid character(s): {invalid_chars!r}'
            raise ValueError(msg)

        return


class Run(Generic[_T_co]):
    """Execution lifecycle for a single run of an Experiment.

    This class is a context manager for the experiment. It encapsulates its
    execution so it can be run preventing conflicts with other experiments or
    runs.

    Attributes:
        run_id: the id of the run.
        metadata_manager: object responsible to register objects to this run.
    """

    def __init__(
            self,
            experiment: Experiment[_T_co],
            *,
            run_id: str | None = None,
            resume_last_run: bool = False,
    ) -> None:
        """Constructor.

        Args:
            experiment: the experiment this run belongs to.
            run_id: identifier of the run, else a timestamp is used.
            resume_last_run: if run_id is not set, resume the previous run.
        """
        self._experiment = experiment
        self.run_id = run_id or experiment.created_at
        self.metadata_manager = tracking.MetadataManager()
        self._resume_last_run = resume_last_run
        self._validate_run_id(run_id, resume_last_run)
        self._experiment.run = self

    def __enter__(self) -> Self:
        """Start the run scope."""
        Experiment.set_current(self.experiment)
        log_events.Event.set_auto_publish(self.experiment.trackers.publish)
        log_events.StartExperimentEvent(
            self.experiment.config,
            self.experiment.name,
            self.experiment.created_at,
            self.run_id,
            self.experiment.par_dir,
            self._resume_last_run,
            self.experiment.tags,
        )
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> None:
        """Conclude the run scope."""
        log_events.StopExperimentEvent(self.experiment.name)
        log_events.Event.set_auto_publish(None)
        Experiment.clear_current()

    @property
    def experiment(self) -> Experiment[_T_co]:
        """The experiment this run belongs to."""
        return self._experiment

    @staticmethod
    def _validate_run_id(run_id: str | None, resume_last_run: bool) -> None:
        if run_id is not None and resume_last_run:
            raise ValueError(
                'Cannot resume last run when a run_id is provided.'
            )

        return

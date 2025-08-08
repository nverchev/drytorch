"""Module containing the experiment class."""

from __future__ import annotations

import pathlib

from types import TracebackType
from typing import Any, Final, Generic, Self, TypeVar

from drytorch.core import exceptions, log_events, tracking
from drytorch.utils import repr_utils


_T_co = TypeVar('_T_co', covariant=True)
_U_co = TypeVar('_U_co', covariant=True)


class Experiment(repr_utils.Versioned, Generic[_T_co]):
    """Manage experiment configuration, metadata, and tracking.

    This class links code related to a machine learning experiment to a
    configuration object. Specifically:
        - it makes the configuration globally accessible
        - prevents conflicts between experiments
        - it standardizes the settings across external trackers

    Attributes:
        par_dir: parent directory for experiment data.
        tags: descriptors for the experiment's branch.
        trackers: dispatcher for publishing events.
    """

    _current: Experiment[Any] | None = None
    _name = repr_utils.DefaultName()

    def __init__(
            self,
            config: _T_co,
            *,
            name: str = '',
            par_dir: str | pathlib.Path = pathlib.Path(),
            run_id: str | None = None,
            resume_last_run: bool = False,
            tags: list[str] | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: the name of the experiment. Defaults to class name.
            par_dir: parent directory for experiment data.
            run_id: identifier of the run, else a timestamp is used.
            config: configuration for the experiment.
            resume_last_run: if run_id is not set, resume the previous run.
            tags: descriptors for the experiment's branch (e.g., "lr=0.01").
        """
        super().__init__()
        self.__config: Final[_T_co] = config
        self._name = name
        self.par_dir = pathlib.Path(par_dir)
        self.run_id: str = run_id or self.created_at
        self._resume_last_run = resume_last_run
        self.tags = tags or []
        self.trackers = tracking.EventDispatcher(self.name)
        self.trackers.register(**tracking.DEFAULT_TRACKERS)
        self._metadata_manager = tracking.MetadataManager()
        self._validate_name(name)
        self._validate_run_id(run_id, resume_last_run)
        return

    def __enter__(self) -> Self:
        """Start the experiment scope."""
        current_exp = Experiment._current
        if current_exp is not None:
            raise exceptions.NestedScopeError(current_exp.name, self.name)
        Experiment._current = self
        log_events.Event.set_auto_publish(self.trackers.publish)
        log_events.StartExperimentEvent(
            self.__config,
            self.name,
            self.created_at,
            self.run_id,
            self.par_dir,
            self._resume_last_run,
            self.tags,
        )
        return self

    def __exit__(
            self,
            exc_type: type[BaseException],
            exc_val: BaseException,
            exc_tb: TracebackType,
    ) -> None:
        """Conclude the experiment scope."""
        log_events.StopExperimentEvent(self.name)
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
        """
        return cls.current().__config

    @classmethod
    def _check_if_active(cls) -> bool:
        return isinstance(Experiment.current(), cls)

    @staticmethod
    def _validate_name(name: str) -> None:
        not_allowed_chars = set(r'\/:*?"<>|')
        if invalid_chars := set(name) & not_allowed_chars:
            msg = f'Name contains invalid character(s): {invalid_chars!r}'
            raise ValueError(msg)

        return

    @staticmethod
    def _validate_run_id(run_id: str | None, resume_last_run: bool) -> None:
        if run_id is not None and resume_last_run:
            raise ValueError(
                'Cannot resume last run when a run_id is provided.'
            )

        return

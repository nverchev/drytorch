"""Experiment and Run classes."""

from __future__ import annotations

import dataclasses
import datetime
import gc
import importlib.metadata
import json
import multiprocessing
import pathlib
import shutil
import subprocess
import time
import types
import warnings
import weakref

from collections.abc import Mapping, Sequence
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Self,
    TypeVar,
)

import filelock

from torch import distributed as dist
from typing_extensions import override

from drytorch.core import exceptions, log_events, tracking
from drytorch.utils import repr_utils


__all__ = [
    'Experiment',
    'Run',
]

_T_co = TypeVar('_T_co', covariant=True)
RunStatus = Literal['created', 'running', 'completed', 'failed', 'paused']


@dataclasses.dataclass
class RunSession:
    """A start or resumption of a run.

    Attributes:
        timestamp: when the session started.
        commit: git commit hash of the code, if available.
        drytorch_version: the drytorch version in use.
    """

    timestamp: str
    commit: str | None
    drytorch_version: str


@dataclasses.dataclass
class RunMetadata:
    """Metadata for a run.

    Attributes:
        id: unique identifier for the run.
        status: status of the run.
        sessions: the start of the run and each resumption, in order.
    """

    id: str
    status: RunStatus
    sessions: list[RunSession] = dataclasses.field(default_factory=list)


class RunRegistry:
    """Creates and manages a JSON file for run metadata.

    Attributes:
        file_path: path to the JSON file.
        lock_path: path to the lock file for file operations.
    """

    file_path: Final[pathlib.Path]
    lock_path: Final[pathlib.Path]

    def __init__(self, path: pathlib.Path):
        """Initialize.

        Args:
            path: path to the JSON file.
        """
        self.file_path = path
        self.lock_path = path.with_suffix('.json.lock')
        return

    def load_all(self) -> list[RunMetadata]:
        """Loads all run metadata from a JSON file."""
        if not self.file_path.exists():
            return []

        try:
            with self.file_path.open() as f:
                data = json.load(f)

        except (FileNotFoundError, json.JSONDecodeError):
            return []

        recorded_metadata: list[RunMetadata] = []
        for item in data:
            sessions = [RunSession(**s) for s in item.get('sessions', [])]
            metadata = RunMetadata(item['id'], item['status'], sessions)
            recorded_metadata.append(metadata)

        return recorded_metadata

    def add_session(self, run_id: str, session: RunSession) -> None:
        """Record a start or resumption of a run.

        Args:
            run_id: the id of the run.
            session: the new session.

        Raises:
            RunNotRecordedError: if the run is not in the registry.
        """
        with filelock.FileLock(self.lock_path):
            recorded_metadata = self.load_all()
            for metadata in recorded_metadata:
                if metadata.id == run_id:
                    metadata.sessions.append(session)
                    break
            else:
                raise exceptions.RunNotRecordedError(run_id)

            self._save(recorded_metadata)

        return

    def register_new_run(self, run_metadata: RunMetadata) -> None:
        """Register a new run, ensuring a unique ID.

        Args:
            run_metadata: the metadata of the new run.

        Raises:
            RunAlreadyRecordedError: if the run ID is already registered.
        """
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(self.lock_path):
            recorded_metadata = self.load_all()
            if any(m.id == run_metadata.id for m in recorded_metadata):
                raise exceptions.RunAlreadyRecordedError(
                    run_metadata.id, self.file_path.parent.name
                )

            recorded_metadata.append(run_metadata)
            self._save(recorded_metadata)

        return

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        """Update the status of a run.

        Args:
            run_id: the id of the run to update.
            status: the new status.

        Raises:
            RunNotRecordedError: if the run is not in the registry.
        """
        with filelock.FileLock(self.lock_path):
            recorded_metadata = self.load_all()
            for metadata in recorded_metadata:
                if metadata.id == run_id:
                    metadata.status = status
                    break
            else:
                raise exceptions.RunNotRecordedError(run_id)

            self._save(recorded_metadata)

        return

    def _save(self, recorded_metadata: list[RunMetadata]) -> None:
        serialized = [dataclasses.asdict(m) for m in recorded_metadata]
        self.file_path.write_text(json.dumps(serialized, indent=2))
        return


class Experiment(Generic[_T_co]):
    """Manage experiment configuration, directory, and tracking.

    This class associates a configuration file, a name, and a working directory
    with a machine learning experiment. It also contains the trackers
    responsible for tracking the metadata and metrics for the experiment.
    Finally, it allows global access to a configuration file with the correct
    type annotations.

    Attributes:
        par_dir: parent directory for experiment data.
        tags: descriptors for the experiment.
        trackers: dispatcher for publishing events.
        folder_name: name of the hidden folder storing experiment metadata.
        run_file: filename storing the registry of run IDs for this experiment.
    """

    folder_name: ClassVar[str] = '.drytorch'
    run_file: ClassVar[str] = 'runs.json'
    _name = repr_utils.DefaultName()
    __current: ClassVar[Experiment[Any] | None] = None
    par_dir: Final[pathlib.Path]
    tags: Final[Sequence[str]]
    trackers: Final[tracking.EventDispatcher]
    _active_run: Run[_T_co] | None
    _paused_runs: dict[str, Run[_T_co]]
    _registry: RunRegistry
    __config: Final[_T_co]

    def __init__(
        self,
        config: _T_co,
        *,
        name: str = '',
        par_dir: str | pathlib.Path = pathlib.Path(),
        tags: Sequence[str] | None = None,
    ) -> None:
        """Initialize.

        Args:
            config: configuration for the experiment.
            name: the name of the experiment (defaults to class name).
            par_dir: parent directory for experiment data.
            tags: descriptors for the experiment (e.g., ``"lr=0.01"``).
        """
        _validate_chars(name)
        self.__config = config
        self._name = name
        self.par_dir = pathlib.Path(par_dir)
        self.tags = tuple(tags) if tags else ()
        self.trackers = tracking.EventDispatcher(self.name)
        self.trackers.subscribe(**tracking.DEFAULT_TRACKERS)
        run_file = self.par_dir / self.folder_name / self.name / self.run_file
        self._registry = RunRegistry(run_file)
        self._active_run = None
        self._paused_runs = {}
        return

    @property
    def config(self) -> _T_co:
        """Experiment configuration."""
        return self.__config

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return self._name

    @property
    def paused_runs(self) -> Mapping[str, Run[_T_co]]:
        """A snapshot of runs paused in this session."""
        return dict(self._paused_runs)

    @property
    def run(self) -> Run[_T_co]:
        """Get the current run.

        Raises:
            NoActiveExperimentError: if no run is currently active.
        """
        if self._active_run is None:
            raise exceptions.NoActiveExperimentError(self.name)

        return self._active_run

    @override
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    def create_run(
        self,
        *,
        record: bool = True,
        resume: bool = False,
        run_id: str | None = None,
    ) -> Run[_T_co]:
        """Convenience constructor for a Run using this experiment.

        Args:
            record: record the run in the registry.
            resume: resume the selected run if run_id is set, else the last run.
            run_id: identifier of the run; defaults to timestamp.

        Returns:
            The created run object.

        Raises:
            RunAlreadyRecordedError: if creating a new run with an existing id.
            RunStillPausedError: if creating a run with the id of a paused run.
        """
        if run_id is not None:
            _validate_chars(run_id)
            if run_id in self._paused_runs:
                raise exceptions.RunStillPausedError(run_id, self.name)

        record = record and _is_main_process()
        recorded_metadata = self._registry.load_all()
        if resume:
            return self._handle_resume_logic(run_id, recorded_metadata, record)

        if run_id in [m.id for m in recorded_metadata]:
            raise exceptions.RunAlreadyRecordedError(run_id, self.name)

        return self._create_new_run(run_id, record)

    def _handle_resume_logic(
        self,
        run_id: str | None,
        recorded_metadata: list[RunMetadata],
        record: bool,
    ) -> Run[_T_co]:
        """Handle resume logic for existing runs."""
        if not recorded_metadata:
            warnings.warn(exceptions.NoPreviousRunsWarning(), stacklevel=2)
            return self._create_new_run(run_id, record)

        if run_id is None:
            run_id = recorded_metadata[-1].id
        else:
            matching_metadata = [m for m in recorded_metadata if m.id == run_id]
            if not matching_metadata:
                warnings.warn(
                    exceptions.NotExistingRunWarning(run_id), stacklevel=1
                )
                return self._create_new_run(run_id, record)

            if len(matching_metadata) > 1:
                msg = f'Multiple runs with id {run_id} found in the registry.'
                raise RuntimeError(msg)

        if run_id in self._paused_runs:
            raise exceptions.RunStillPausedError(run_id, self.name)

        return Run(self, run_id, record=record, resumed=True)

    def _create_new_run(
        self,
        run_id: str | None,
        record: bool,
    ) -> Run[_T_co]:
        """Create a new run (non-resume case)."""
        new_id = self._make_run_id(run_id)
        while record and not self._try_register(new_id):
            if run_id is not None:
                raise exceptions.RunAlreadyRecordedError(new_id, self.name)

            time.sleep(1)  # the next timestamp gives a new id
            new_id = self._make_run_id(None)

        return Run(self, new_id, record=record)

    def _try_register(self, run_id: str) -> bool:
        """Record a new run; return False if its id is already recorded."""
        run_metadata = RunMetadata(id=run_id, status='created')
        try:
            self._registry.register_new_run(run_metadata)
        except exceptions.RunAlreadyRecordedError:
            return False

        return True

    @classmethod
    def get_config(cls) -> _T_co:
        """Retrieve the configuration of the current experiment."""
        return cls.get_current().__config

    @classmethod
    def get_current(cls) -> Self:
        """Return the currently active experiment.

        Raises:
            NoActiveExperimentError: if no experiment is currently active.
        """
        current = Experiment.__current
        if current is None:
            raise exceptions.NoActiveExperimentError()

        if not isinstance(current, cls):
            raise exceptions.NoActiveExperimentError(experiment_class=cls)

        return current

    @classmethod
    def has_current(cls) -> bool:
        """Check whether there is an active experiment."""
        return Experiment.__current is not None

    @staticmethod
    def set_current(experiment: Experiment[Any]) -> None:
        """Set an experiment as active.

        Raises:
            NestedScopeError: if there is an already active run.
        """
        if (old_exp := Experiment.__current) is not None:
            raise exceptions.NestedScopeError(old_exp.name, experiment.name)

        Experiment.__current = experiment
        return

    @staticmethod
    def _clear_current() -> None:
        """Clear the active experiment."""
        Experiment.__current = None
        return

    @staticmethod
    def _make_run_id(run_id: str | None) -> str:
        """Return the id, or a timestamp; worker processes append their PID."""
        final_id = run_id or _timestamp()
        if not _is_distributed():  # keep the same ID for distributed runs
            if multiprocessing.current_process().name != 'MainProcess':
                final_id = f'{final_id}_{multiprocessing.current_process().pid}'

        return final_id


class Run(repr_utils.CreatedAtMixin, Generic[_T_co]):
    """Execution lifecycle for a single run of an Experiment.

    Attributes:
        metadata_manager: Manager for run metadata.
        record: whether to record the run in the registry.
        resumed: whether the run was resumed.
        status: Current status of the run.
    """

    _STATUS_WARNINGS: ClassVar[
        dict[RunStatus, type[exceptions.DryTorchWarning]]
    ] = {
        'created': exceptions.RunNotStartedWarning,
        'running': exceptions.RunAlreadyRunningWarning,
        'paused': exceptions.RunAlreadyPausedWarning,
        'completed': exceptions.RunAlreadyCompletedWarning,
        'failed': exceptions.RunAlreadyFailedWarning,
    }

    metadata_manager: Final[tracking.MetadataManager]
    record: Final[bool]
    resumed: Final[bool]
    status: RunStatus
    _experiment: Final[Experiment[_T_co]]
    _finalizer: weakref.finalize[..., Self] | None
    _id: Final[str]

    def __init__(
        self,
        experiment: Experiment[_T_co],
        run_id: str,
        *,
        record: bool = True,
        resumed: bool = False,
    ) -> None:
        """Initialize.

        Args:
            experiment: the experiment this run belongs to.
            run_id: identifier of the run.
            record: record the run in the registry.
            resumed: whether the run was resumed.
        """
        super().__init__()
        self.metadata_manager = tracking.MetadataManager()
        self.record = record
        self.resumed = resumed
        self.status = 'created'
        self._experiment = experiment
        self._finalizer = None
        self._id = run_id
        return

    @property
    def experiment(self) -> Experiment[_T_co]:
        """The experiment this run belongs to."""
        return self._experiment

    @property
    def id(self) -> str:
        """The identifier of the run."""
        return self._id

    def __enter__(self) -> Self:
        """Enter the experiment scope."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the experiment scope, leaving a paused run paused."""
        if exc_type is not None:
            if self.status in ('running', 'paused'):
                self.status = 'failed'
                self._close()
            return

        if self.status == 'paused':
            return

        self.stop()
        return

    @override
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(id={self.id}, status={self.status})'

    def is_active(self) -> bool:
        """Check if the run is currently active."""
        return self.status == 'running'

    def pause(self) -> None:
        """Pause the experiment scope."""
        if self.status != 'running':
            warnings.warn(self._STATUS_WARNINGS[self.status](), stacklevel=1)
            return

        self.status = 'paused'
        if self.record:
            self._update_registry()

        self._experiment._paused_runs[self._id] = self
        self._pause_experiment(self.experiment, self._id)
        return

    def start(self) -> None:
        """Start the experiment scope."""
        if self.status not in ('created', 'paused'):
            warnings.warn(self._STATUS_WARNINGS[self.status](), stacklevel=1)
            return

        continuing = self.status == 'paused'
        Experiment.set_current(self._experiment)
        self.status = 'running'
        if self.record:
            self._update_registry()
            if not continuing:
                registry = self._experiment._registry
                registry.add_session(self._id, _new_session())

        self._experiment._active_run = self
        if self._experiment._paused_runs.get(self._id) is self:
            del self._experiment._paused_runs[self._id]

        if self._finalizer is not None:
            self._finalizer.detach()

        self._finalizer = weakref.finalize(
            self, self._stop_experiment, self._experiment, self._id
        )

        if _is_main_process():
            log_events.Event.set_auto_publish(self._experiment.trackers.publish)
        else:  # no tracking in secondary processes
            log_events.Event.set_auto_publish(lambda _: None)

        if continuing:
            log_events.ContinueExperimentEvent(self._experiment.name, self._id)
        else:
            log_events.StartExperimentEvent(
                self._experiment.config,
                self._experiment.name,
                self.created_at,
                self._id,
                self.resumed,
                self._experiment.par_dir,
                self._experiment.tags,
            )
        return

    def stop(self) -> None:
        """Stop the experiment scope."""
        if self.status not in ('running', 'paused'):
            warnings.warn(self._STATUS_WARNINGS[self.status](), stacklevel=1)
            return

        self.status = 'completed'
        self._close()
        return

    def _close(self) -> None:
        if self.record:
            self._update_registry()

        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None

        if self._experiment._paused_runs.get(self._id) is self:
            del self._experiment._paused_runs[self._id]

        self._stop_experiment(self.experiment, self._id)
        return

    def _update_registry(self) -> None:
        """Update the run status in the experiment's registry."""
        self.experiment._registry.update_run_status(self.id, self.status)
        return

    @staticmethod
    def _pause_experiment(experiment: Experiment[Any], run_id: str) -> None:
        """Cleanup on pause without holding reference to a Run instance."""
        log_events.PauseExperimentEvent(experiment.name, run_id)
        Run._teardown(experiment)
        return

    @staticmethod
    def _stop_experiment(
        experiment: Experiment[Any],
        run_id: str,
    ) -> None:
        """Cleanup without holding reference to a Run instance."""
        active_run = experiment._active_run
        if active_run is not None:
            if active_run.id == run_id:
                log_events.StopExperimentEvent(experiment.name, run_id)
                Run._teardown(experiment)

            return

        if not Experiment.has_current():
            experiment.trackers.close_all()

        return

    @staticmethod
    def _teardown(experiment: Experiment[Any]) -> None:
        """Common cleanup for pause and stop."""
        log_events.Event.set_auto_publish(None)
        experiment._active_run = None
        Experiment._clear_current()
        gc.collect()
        return


def _get_last_commit_hash() -> str | None:
    git = shutil.which('git')
    if git is None:
        return None

    try:
        result = subprocess.run(  # noqa: S603
            [git, 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    return result.stdout.strip()


def _is_distributed() -> bool:
    """Whether this process belongs to an initialized distributed group."""
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    """Whether this process is not distributed or has rank 0."""
    return not _is_distributed() or dist.get_rank() == 0


def _new_session() -> RunSession:
    return RunSession(
        timestamp=_timestamp(),
        commit=_get_last_commit_hash(),
        drytorch_version=importlib.metadata.version('drytorch'),
    )


def _timestamp() -> str:
    """Return the current time in the format used for run ids."""
    return datetime.datetime.now().strftime(repr_utils.CreatedAtMixin.ts_fmt)


def _validate_chars(name: str) -> None:
    if len(name) > 255:
        msg = f'Name is too long (max 255 chars): {len(name)}'
        raise ValueError(msg)

    not_allowed_chars = set(r'\/:*?"<>|')
    if invalid_chars := set(name) & not_allowed_chars:
        msg = f'Name contains invalid character(s): {invalid_chars!r}'
        raise ValueError(msg)

    return

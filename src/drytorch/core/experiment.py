"""Module containing the Experiment and Run class."""

from __future__ import annotations

import dataclasses
import gc
import json
import pathlib
import warnings
import weakref

from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Self,
    TypeVar,
)
from weakref import finalize

from typing_extensions import override

from drytorch.core import exceptions, log_events, track
from drytorch.utils import repr_utils


_T_co = TypeVar('_T_co', covariant=True)

RunStatus = Literal['created', 'running', 'completed', 'failed']


@dataclasses.dataclass
class RunMetadata:
    """Metadata for a run."""

    id: str
    status: RunStatus
    timestamp: str


class RunRegistry:
    """Creates and manages a JSON file for run metadata.

    Attributes:
        file_path: path to the JSON file.
    """

    def __init__(self, path: pathlib.Path):
        """Constructor.

        Args:
            path: path to the JSON file.
        """
        self.file_path: Path = path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.save_all([])

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

        run_data = []
        for item in data:
            run_data.append(RunMetadata(**item))

        return run_data

    def save_all(self, runs: list[RunMetadata]) -> None:
        """Saves all run metadata to a JSON file."""
        run_data = []
        for run in runs:
            run_data.append(dataclasses.asdict(run))

        self.file_path.write_text(json.dumps(run_data, indent=2))


class Experiment(Generic[_T_co]):
    """Manage experiment configuration, directory, and tracking.

    This class associates a configuration file, a name, and a working directory
    with a machine learning experiment. It also contains the trackers
    responsible for tracking the metadata and metrics for the experiment.
    Finally, it allows global access to a configuration file with the correct
    type annotations.

    Class Variables:
        runs: a list of all previous runs created by this class.
        folder_name: name of the hidden folder storing experiment metadata.
        run_file: filename storing the registry of run IDs for this experiment.

    Attributes:
        par_dir: parent directory for experiment data.
        tags: descriptors for the experiment.
        trackers: dispatcher for publishing events.
    """

    _name = repr_utils.DefaultName()
    __current: ClassVar[Experiment[Any] | None] = None
    folder_name: ClassVar[str] = '.drytorch'
    run_file: ClassVar[str] = 'runs.json'

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
            config: Configuration for the experiment.
            name: The name of the experiment (defaults to class name).
            par_dir: Parent directory for experiment data.
            tags: Descriptors for the experiment (e.g., ``"lr=0.01"``).
        """
        _validate_chars(name)
        self.__config: Final[_T_co] = config
        self._name = name
        self.par_dir: Path = pathlib.Path(par_dir)
        self.tags: list[str] = tags or []
        self.trackers: Final = track.EventDispatcher(self.name)
        self.trackers.register(**track.DEFAULT_TRACKERS)
        run_file = self.par_dir / self.folder_name / self.name / self.run_file
        self._registry: RunRegistry = RunRegistry(run_file)
        self._active_run: Run[_T_co] | None = None
        self.previous_runs: list[Run[_T_co]] = []

    @override
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return self._name

    @property
    def config(self) -> _T_co:
        """Experiment configuration."""
        return self.__config

    def create_run(
        self,
        *,
        run_id: str | None = None,
        resume: bool = False,
        record: bool = True,
    ) -> Run[_T_co]:
        """Convenience constructor for a Run using this experiment.

        Args:
            run_id: identifier of the run; defaults to timestamp.
            resume: resume the selected run if run_id is set, else the last run.
            record: register the run in the registry.

        Returns:
            Run: The created run object.
        """
        if run_id is not None:
            _validate_chars(run_id)

        runs_data = self._registry.load_all()
        if resume:
            return self._handle_resume_logic(run_id, runs_data, record)
        else:
            return self._create_new_run(run_id, runs_data, record)

    def _handle_resume_logic(
        self, run_id: str | None, runs_data: list[RunMetadata], record: bool
    ) -> Run[_T_co]:
        """Handle resume logic for existing runs."""
        if self.previous_runs:
            run = self._get_run_from_previous(run_id)
            if run:
                run.resumed = True
                run.status = 'created'
                return run

        if not runs_data:
            warnings.warn(exceptions.NoPreviousRunsWarning(), stacklevel=2)
            return self._create_new_run(run_id, runs_data, record)

        if run_id is None:
            run_id = runs_data[-1].id
        else:
            matching_runs = [r for r in runs_data if r.id == run_id]
            if not matching_runs:
                warnings.warn(
                    exceptions.NotExistingRunWarning(run_id), stacklevel=1
                )
                return self._create_new_run(run_id, runs_data, record)

            if len(matching_runs) > 1:
                msg = f'Multiple runs with id {run_id} found in the registry.'
                raise RuntimeError(msg)

        return Run(experiment=self, run_id=run_id, resumed=True, record=record)

    def _get_run_from_previous(self, run_id: str | None) -> Run[_T_co] | None:
        """Get run from the previous_runs list."""
        if run_id is None:
            return self.previous_runs[-1]

        matching_runs = [r for r in self.previous_runs if r.id == run_id]
        if not matching_runs:
            return None

        matching_run, *other_runs = matching_runs
        if other_runs:
            msg = f'Multiple runs with id {run_id} found for exp {self.name}'
            raise RuntimeError(msg)

        return matching_run

    def _create_new_run(
        self,
        run_id: str | None,
        runs_data: list[RunMetadata],
        record: bool,
    ) -> Run[_T_co]:
        """Create a new run (non-resume case)."""
        run = Run(experiment=self, run_id=run_id, record=record)
        run_data = RunMetadata(
            id=run.id,
            status='created',
            timestamp=run.created_at_str,
        )
        runs_data.append(run_data)
        if record:
            self._registry.save_all(runs_data)

        return run

    @property
    def run(self) -> Run[_T_co]:
        """Get the current run."""
        if self._active_run is None:
            raise exceptions.NoActiveExperimentError(self.name)
        return self._active_run

    @run.setter
    def run(self, current_run: Run[_T_co]) -> None:
        self._active_run = current_run
        return

    @classmethod
    def get_config(cls) -> _T_co:
        """Retrieve the configuration of the current experiment."""
        return cls.get_current().__config

    @classmethod
    def get_current(cls) -> Self:
        """Return the currently active experiment."""
        if Experiment.__current is None:
            raise exceptions.NoActiveExperimentError()

        if not isinstance(Experiment.__current, cls):
            raise exceptions.NoActiveExperimentError(experiment_class=cls)

        return Experiment.__current

    @staticmethod
    def set_current(experiment: Experiment[_T_co]) -> None:
        """Set an experiment as active."""
        if (old_exp := Experiment.__current) is not None:
            raise exceptions.NestedScopeError(old_exp.name, experiment.name)

        Experiment.__current = experiment
        return

    @staticmethod
    def _clear_current() -> None:
        """Clear the active experiment."""
        Experiment.__current = None
        return


class Run(repr_utils.CreatedAtMixin, Generic[_T_co]):
    """Execution lifecycle for a single run of an Experiment.

    Attributes:
        status: Current status of the run.
        resumed: whether the run was resumed.
        metadata_manager: Manager for run metadata.
        record: whether to record the run in the registry.
    """

    def __init__(
        self,
        experiment: Experiment[_T_co],
        run_id: str | None,
        resumed: bool = False,
        record: bool = True,
    ) -> None:
        """Constructor.

        Args:
            experiment: the experiment this run belongs to.
            run_id: identifier of the run.
            resumed: whether the run was resumed.
            record: register the run in the registry.
        """
        super().__init__()
        self._experiment: Final[Experiment[_T_co]] = experiment
        self._id: Final = run_id or self.created_at_str
        self.resumed: bool = resumed
        self.record: bool = record
        self.status: RunStatus = 'created'
        self.metadata_manager: Final = track.MetadataManager()
        self._finalizer: finalize[..., Self] | None = None
        if not self.resumed:
            experiment.previous_runs.append(self)

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
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the experiment scope."""
        if exc_type is not None:
            self.status = 'failed'

        self.stop()
        return

    def is_active(self) -> bool:
        """Check if the run is currently active."""
        return self.status == 'running'

    def stop(self) -> None:
        """Stop the experiment scope."""
        if self.status == 'running':  # failed is left as is
            self.status = 'completed'
        elif self.status == 'created':
            warnings.warn(exceptions.RunNotStartedWarning(), stacklevel=1)
            return
        elif self.status == 'completed':
            warnings.warn(exceptions.RunAlreadyCompletedWarning(), stacklevel=1)
            return
        if self.record:
            self._update_registry()
        self._cleanup_resources(self.experiment)
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        return

    def start(self: Self) -> None:
        """Start the experiment scope."""
        if self.status == 'running':
            warnings.warn(exceptions.RunAlreadyRunningWarning(), stacklevel=1)
            return
        self._finalizer = weakref.finalize(
            self, self._cleanup_resources, self._experiment
        )
        self.status = 'running'
        if self.record:
            self._update_registry()
        self._experiment._active_run = self
        Experiment.set_current(self._experiment)
        log_events.Event.set_auto_publish(self._experiment.trackers.publish)
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

    def _update_registry(self) -> None:
        """Update the run status in the experiment's registry."""
        run_data = self.experiment._registry.load_all()

        for run_metadata in run_data:
            if run_metadata.id == self.id:
                run_metadata.status = self.status
                break
        else:
            run_data.append(
                RunMetadata(
                    id=self.id,
                    status=self.status,
                    timestamp=self.created_at_str,
                )
            )

        self.experiment._registry.save_all(run_data)

        return

    @staticmethod
    def _cleanup_resources(experiment: Experiment[_T_co]) -> None:
        """Cleanup without holding reference to a Run instance."""
        experiment._active_run = None
        log_events.StopExperimentEvent(experiment.name)
        log_events.Event.set_auto_publish(None)
        Experiment._clear_current()
        gc.collect()
        return


def _validate_chars(name: str) -> None:
    if len(name) > 255:
        msg = f'Name is too long (max 255 chars): {len(name)}'
        raise ValueError(msg)

    not_allowed_chars = set(r'\/:*?"<>|')
    if invalid_chars := set(name) & not_allowed_chars:
        msg = f'Name contains invalid character(s): {invalid_chars!r}'
        raise ValueError(msg)

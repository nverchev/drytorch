"""Module containing the Experiment and Run class."""

from __future__ import annotations

import dataclasses
import json
import pathlib

from types import TracebackType
from typing import Any, ClassVar, Final, Generic, Literal, Self, TypeVar

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
    hashed_config: int | None = None


class RunIO:
    """Creates and manages a JSON file for run metadata.

    Attributes:
        json_file: path to the JSON file.
    """

    def __init__(self, path: pathlib.Path):
        """Constructor.

        Args:
            path: path to the JSON file.
        """
        self.json_file = path
        self.json_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.json_file.exists():
            self.save_all([])

        return

    def load_all(self) -> list[RunMetadata]:
        """Loads all run metadata from a JSON file."""
        if not self.json_file.exists():
            return []

        try:
            with self.json_file.open() as f:
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

        self.json_file.write_text(json.dumps(run_data, indent=2))


class Experiment(repr_utils.CreatedAtMixin, Generic[_T_co]):
    """Manage experiment configuration, directory, and tracking.

    This class associates a configuration file, a name, and a working directory
    with a machine learning experiment. It also contains the trackers
    responsible for tracking the metadata and metrics for the experiment.
    Finally, it allows global access to a configuration file with the correct
    type annotations.

    Class Variables:
        runs: List of all previous runs created by this class.
        folder_name: Name of the hidden folder storing experiment metadata.
        run_file: Filename storing the registry of run IDs for this experiment.

    Attributes:
        par_dir: Parent directory for experiment data.
        tags: Descriptors for the experiment.
        trackers: Dispatcher for publishing events.
    """

    _name = repr_utils.DefaultName()
    __current: Experiment[Any] | None = None
    folder_name: ClassVar[str] = '.drytorch'
    run_file: ClassVar[str] = 'runs.json'

    def __init__(
            self,
            config: _T_co,
            *,
            name: str = "",
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
        super().__init__()
        _validate_chars(name)
        self.__config: Final[_T_co] = config
        self._name = name
        self.par_dir = pathlib.Path(par_dir)
        self.tags = tags or []
        self.trackers = track.EventDispatcher(self.name)
        self.trackers.register(**track.DEFAULT_TRACKERS)
        run_file = self.par_dir / self.folder_name / self.name / self.run_file
        self.run_io = RunIO(run_file)
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
    ) -> Run[_T_co]:
        """Convenience constructor for a Run using this experiment.

        Args:
            run_id: identifier of the run; defaults to timestamp.
            resume: resume the selected run if run_id is set, else the last run.

        Returns:
            Run: The created run object.
        """
        if run_id is not None:
            _validate_chars(run_id)

        runs_data = self.run_io.load_all()
        if resume:
            return self._handle_resume_logic(run_id, runs_data)
        else:
            return self._create_new_run(run_id, runs_data)

    def _handle_resume_logic(
            self,
            run_id: str | None,
            runs_data: list
    ) -> Run[_T_co]:
        """Handle resume logic for existing runs."""
        if self.previous_runs:
            run = self._get_run_from_previous(run_id)
            if run:
                run.resumed = True
                run.status = 'created'
                return run

        if not runs_data:
            raise ValueError(
                f"No previous runs found for experiment {self.name}"
            )

        resolved_run_id = self._resolve_run_id_from_data(run_id, runs_data)
        return Run(experiment=self, run_id=resolved_run_id, resumed=True)

    def _get_run_from_previous(self, run_id: str | None) -> Run[_T_co] | None:
        """Get run from the previous_runs list."""
        if run_id is None:
            return self.previous_runs[-1]

        matching_runs = [r for r in self.previous_runs if r.id == run_id]
        if not matching_runs:
            return None

        if len(matching_runs) > 1:
            raise ValueError(
                f"Multiple runs with id {run_id} found for "
                f"experiment {self.name}"
            )

        return matching_runs[0]

    def _resolve_run_id_from_data(
            self,
            run_id: str | None,
            runs_data: list
    ) -> str:
        """Resolve run_id from runs_data or validate an existing one."""
        if run_id is None:
            return runs_data[-1].id

        if not any(r_data.id == run_id for r_data in runs_data):
            raise ValueError(
                f"Run {run_id} not found for experiment {self.name}"
            )

        return run_id

    def _create_new_run(
            self,
            run_id: str | None,
            runs_data: list
    ) -> Run[_T_co]:
        """Create a new run (non-resume case)."""
        resolved_run_id = run_id or self.created_at_str
        try:
            hashed_config: int | None = hash(self.__config)
        except TypeError:
            hashed_config = None
        run_data = RunMetadata(id=resolved_run_id,
                               status='created',
                               hashed_config=hashed_config)
        runs_data.append(run_data)
        self.run_io.save_all(runs_data)
        return Run(experiment=self, run_id=resolved_run_id)

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

        # noinspection PyUnreachableCode
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
    def clear_current() -> None:
        """Clear the active experiment."""
        if Experiment.__current is None:
            raise exceptions.NoActiveExperimentError()

        Experiment.__current = None
        return


class Run(Generic[_T_co]):
    """Execution lifecycle for a single run of an Experiment.

    Attributes:
        id: Identifier of the run.
        status: Current status of the run.
        resumed: whether the run was resumed.
        metadata_manager: Manager for run metadata.
    """

    def __init__(
            self,
            experiment: Experiment[_T_co],
            run_id: str,
            resumed: bool = False,
    ) -> None:
        """Constructor.

        Args:
            experiment: the experiment this run belongs to.
            run_id: identifier of the run.
            resumed: whether the run was resumed.
        """
        self._experiment = experiment
        self.id = run_id
        self.resumed = resumed
        self.status: RunStatus = 'created'
        self.metadata_manager = track.MetadataManager()
        experiment.run = self
        if not self.resumed:
            experiment.previous_runs.append(self)

    def __enter__(self) -> Self:
        """Start the experiment scope."""
        self.status = 'running'
        Experiment.set_current(self.experiment)
        log_events.Event.set_auto_publish(self.experiment.trackers.publish)
        log_events.StartExperimentEvent(
            self.experiment.config,
            self.experiment.name,
            self.experiment.created_at,
            self.id,
            self.resumed,
            self.experiment.par_dir,
            self.experiment.tags,
        )
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> None:
        """End the experiment scope."""
        if exc_type is None:
            self.status = 'completed'
        else:
            self.status = 'failed'

        log_events.StopExperimentEvent(self.experiment.name)
        log_events.Event.set_auto_publish(None)
        Experiment.clear_current()

    @property
    def experiment(self) -> Experiment[_T_co]:
        """The experiment this run belongs to."""
        return self._experiment


def _validate_chars(name: str) -> None:
    not_allowed_chars = set(r'\/:*?"<>|')
    if invalid_chars := set(name) & not_allowed_chars:
        msg = f"Name contains invalid character(s): {invalid_chars!r}"
        raise ValueError(msg)

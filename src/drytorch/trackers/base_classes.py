"""Abstract classes for trackers."""

from __future__ import annotations

import abc
import functools
import pathlib

from collections.abc import Generator, Iterable
from typing import ClassVar, Final, Generic, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from typing_extensions import override

from drytorch.core import exceptions, log_events, tracking


__all__ = [
    'BasePlotter',
    'Dumper',
    'HistoryMetric',
    'HistoryMetrics',
    'MemoryMetrics',
    'MetricLoader',
    'NpArray',
    'SourcedArray',
    'SourcedMetric',
    'SourcedMetrics',
]


HistoryMetric: TypeAlias = tuple[list[int], list[float]]
HistoryMetrics: TypeAlias = tuple[list[int], dict[str, list[float]]]
SourcedMetric: TypeAlias = dict[str, HistoryMetric]
SourcedMetrics: TypeAlias = dict[str, HistoryMetrics]
NpArray: TypeAlias = npt.NDArray[np.float64]
SourcedArray: TypeAlias = dict[str, NpArray]

Figure = TypeVar('Figure')


class Dumper(tracking.Tracker):
    """Tracker with a standard folder structure.

    Attributes:
        folder_name: name of the folder containing the output.
        user_par_dir: parent directory for the tracker data.
    """

    folder_name: ClassVar[str] = 'tracker'

    user_par_dir: pathlib.Path | None
    _exp_name: str | None
    _par_dir: pathlib.Path | None
    _run_id: str | None
    _stashed_state: dict[str, tuple[pathlib.Path, str, str]]

    def __init__(self, *, par_dir: pathlib.Path | None = None) -> None:
        """Initialize.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
        """
        super().__init__()
        self.user_par_dir = par_dir
        self._exp_name = None
        self._par_dir = None
        self._run_id = None
        self._stashed_state = {}
        return

    @property
    def exp_name(self) -> str:
        """Return the name of the experiment.

        Raises:
            AccessOutsideScopeError: when the name id not available.
        """
        if self._exp_name is None:
            raise exceptions.AccessOutsideScopeError()

        return self._exp_name

    @property
    def par_dir(self) -> pathlib.Path:
        """Return the parent directory for the experiments.

        Raises:
            AccessOutsideScopeError: when the default folder is not available.
        """
        if self.user_par_dir is None:
            if self._par_dir is None:
                raise exceptions.AccessOutsideScopeError()
            path = self._par_dir
        else:
            path = self.user_par_dir

        return path

    @property
    def run_id(self) -> str:
        """Return the identifier for the experiment run.

        Raises:
            AccessOutsideScopeError: when the id of the run is not available.
        """
        if self._run_id is None:
            raise exceptions.AccessOutsideScopeError()

        return self._run_id

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        self._par_dir = event.par_dir
        self._exp_name = event.exp_name
        self._run_id = event.run_id
        self.par_dir.mkdir(exist_ok=True, parents=True)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.PauseExperimentEvent) -> None:
        self._stashed_state[event.run_id] = (
            self.par_dir,
            self.exp_name,
            self.run_id,
        )
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ContinueExperimentEvent) -> None:
        if event.run_id not in self._stashed_state:
            raise exceptions.NoStashedStateError(self, event.run_id)

        self._par_dir, self._exp_name, self._run_id = self._stashed_state.pop(
            event.run_id
        )
        return super().notify(event)

    @override
    def clean_up(self) -> None:
        """Remove experimental data from the tracker."""
        self._par_dir = None
        self._exp_name = None
        self._run_id = None
        return super().clean_up()

    @override
    def close(self) -> None:
        """Release tracker state, including any paused state."""
        self._stashed_state.clear()
        return super().close()

    def _get_exp_dir(self) -> pathlib.Path:
        exp_dir = self.par_dir / self.folder_name / self.exp_name
        exp_dir.mkdir(exist_ok=True, parents=True)
        return exp_dir

    def _get_run_dir(self, mkdir: bool = True) -> pathlib.Path:
        exp_dir = self._get_exp_dir()
        if '@' in self.run_id:
            day, time = self.run_id.split('@', 1)
            run_dir = exp_dir / day / time
        else:
            run_dir = exp_dir / self.run_id
        if mkdir:
            run_dir.mkdir(exist_ok=True, parents=True)
        return run_dir


class MetricLoader(tracking.Tracker, abc.ABC):
    """Interface for trackers that load metrics."""

    def load_metrics(
        self, model_name: str, max_epoch: int = -1
    ) -> SourcedMetrics:
        """Load metrics from the last run of the experiment.

        Args:
            model_name: the name of the model.
            max_epoch: the maximum epoch to load. Defaults to all.

        Returns:
            The current epochs and named metric values by the source.

        Raises:
            ValueError: if max_epoch is less than -1.
        """
        if max_epoch == 0:
            return {}

        if max_epoch < -1:
            raise ValueError('Max epoch should not be less than -1.')

        return self._load_metrics(model_name, max_epoch)

    @abc.abstractmethod
    def _load_metrics(
        self, model_name: str, max_epoch: int = -1
    ) -> SourcedMetrics: ...


class MemoryMetrics(tracking.Tracker):
    """Keep all metrics in memory.

    Attributes:
        model_dict: all metrics recorded in this session.
    """

    model_dict: Final[dict[str, SourcedMetrics]]
    _metric_loader: Final[MetricLoader | None]

    def __init__(self, *, metric_loader: MetricLoader | None = None) -> None:
        """Initialize.

        Args:
            metric_loader: object to load the metrics.
        """
        super().__init__()
        self._metric_loader = metric_loader
        self.model_dict = {}
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.LoadModelEvent) -> None:
        if self._metric_loader is None:
            return super().notify(event)

        metrics = self._metric_loader.load_metrics(
            event.model_name, event.epoch
        )
        self.model_dict[event.model_name] = metrics
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        sourced_metrics = self.model_dict.setdefault(event.model_name, {})
        epochs, logs_dict = sourced_metrics.setdefault(
            event.source_name, ([], {})
        )
        epochs.append(event.epoch)
        for metric_name, metric_value in event.metrics.items():
            logs_dict.setdefault(metric_name, []).append(metric_value)

        return super().notify(event)


class BasePlotter(MemoryMetrics, Dumper, abc.ABC, Generic[Figure]):
    """Abstract class for plotting the learning curves on request.

    Attributes:
        file_suffix: the suffix of the files the figures are saved to.
    """

    folder_name: ClassVar[str] = 'plots'
    file_suffix: ClassVar[str]

    def __init__(
        self,
        *,
        metric_loader: MetricLoader | None = None,
        par_dir: pathlib.Path | None = None,
    ) -> None:
        """Initialize.

        Args:
            metric_loader: a tracker that can load metrics from a previous run.
            par_dir: the parent directory for the saved figures. Default uses
                the same of the current experiment.
        """
        super().__init__(metric_loader=metric_loader)
        self.user_par_dir = par_dir
        return

    def create_figures(
        self,
        model_name: str,
        *,
        metric_names: Iterable[str] = (),
        source_names: Iterable[str] = (),
        start_epoch: int = 1,
    ) -> dict[str, Figure]:
        """Create a figure for each metric without rendering it.

        Args:
            model_name: the name of the model to plot.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            start_epoch: the epoch from which to start plotting.

        Returns:
            The figures by metric name.

        Raises:
            ValueError: if start_epoch is less than 1.
            ValueError: if the model is not found.
            ValueError: if no metric is left to plot.
        """
        metric_arrays = self._get_metric_arrays(
            model_name, metric_names, source_names, start_epoch
        )
        return {
            metric_name: self._create_figure(
                model_name, metric_name, sourced_array
            )
            for metric_name, sourced_array in metric_arrays.items()
        }

    def plot(
        self,
        model_name: str,
        *,
        metric_names: Iterable[str] = (),
        source_names: Iterable[str] = (),
        start_epoch: int = 1,
    ) -> None:
        """Render the figures of the metrics together.

        Args:
            model_name: the name of the model to plot.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            start_epoch: the epoch from which to start plotting.

        Raises:
            ValueError: if start_epoch is less than 1.
            ValueError: if the model is not found.
            ValueError: if no metric is left to plot.
        """
        figures = self.create_figures(
            model_name,
            metric_names=metric_names,
            source_names=source_names,
            start_epoch=start_epoch,
        )
        self._render(list(figures.values()))
        return

    def save_figures(
        self,
        model_name: str,
        *,
        metric_names: Iterable[str] = (),
        source_names: Iterable[str] = (),
        start_epoch: int = 1,
    ) -> None:
        """Save the figure of each metric in the folder of the current run.

        Args:
            model_name: the name of the model to plot.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            start_epoch: the epoch from which to start plotting.

        Raises:
            AccessOutsideScopeError: if no run is active.
            ValueError: if start_epoch is less than 1.
            ValueError: if the model is not found.
            ValueError: if no metric is left to plot.

        Side Effects:
            Writes one file for each metric in a folder named after the model.
        """
        figures = self.create_figures(
            model_name,
            metric_names=metric_names,
            source_names=source_names,
            start_epoch=start_epoch,
        )
        model_dir = self._get_run_dir() / model_name
        model_dir.mkdir(exist_ok=True)
        for metric_name, figure in figures.items():
            self._save_figure(
                figure, model_dir / f'{metric_name}{self.file_suffix}'
            )

        return

    @abc.abstractmethod
    def _create_figure(
        self, model_name: str, metric_name: str, sourced_array: SourcedArray
    ) -> Figure:
        """Create the figure of a metric from its sourced arrays."""

    def _get_metric_arrays(
        self,
        model_name: str,
        metric_names: Iterable[str],
        source_names: Iterable[str],
        start_epoch: int,
    ) -> dict[str, SourcedArray]:
        if start_epoch < 1:
            raise ValueError('Start epoch must be positive.')

        sourced_metrics = self.model_dict.get(model_name, {})
        if not sourced_metrics and self._metric_loader is not None:
            sourced_metrics = self._metric_loader.load_metrics(model_name)

        if sourced_metrics:
            self.model_dict[model_name] = sourced_metrics
        else:
            msg = f'No model named {model_name} has been found.'
            raise ValueError(msg)

        if source_names:
            sourced_metrics = {
                source: sourced_metrics[source]
                for source in source_names
                if source in sourced_metrics
            }

        if not metric_names:
            all_metrics: Generator[set[str], None, None] = (
                set(logs[1]) for logs in sourced_metrics.values()
            )
            metric_names = sorted(set().union(*all_metrics))

        metric_arrays: dict[str, SourcedArray] = {}
        for metric_name in metric_names:
            processed_sources = self._process_source(
                sourced_metrics, metric_name, start_epoch
            )
            if processed_sources:
                metric_arrays[metric_name] = processed_sources

        if not metric_arrays:
            msg = f'No metric to plot for model {model_name}.'
            raise ValueError(msg)

        return metric_arrays

    def _get_sourced_metric(
        self, sourced_metrics: SourcedMetrics, metric_name: str
    ) -> SourcedMetric:
        out: SourcedMetric = {}
        for source_name, (epochs, metrics) in sourced_metrics.items():
            if epochs and metric_name in metrics:
                values = metrics[metric_name]
                if len(epochs) != len(values):
                    msg = '{} and {} logs refer to different epochs.'
                    msg = msg.format(source_name, metric_name)
                    raise exceptions.TrackerError(self, msg)

                out[source_name] = (epochs, values)

        return out

    def _process_source(
        self, sourced_metrics: SourcedMetrics, metric_name: str, start: int
    ) -> SourcedArray:
        sourced_metric = self._get_sourced_metric(sourced_metrics, metric_name)
        ordered_sources = self._order_sources(sourced_metric)
        sourced_array = self._source_to_numpy(ordered_sources)
        return self._filter_by_epoch(sourced_array, start)

    @abc.abstractmethod
    def _render(self, figures: list[Figure]) -> None:
        """Render the figures together."""

    @abc.abstractmethod
    def _save_figure(self, figure: Figure, path: pathlib.Path) -> None:
        """Save the figure to the path."""

    @classmethod
    def _order_sources(cls, sources: SourcedMetric) -> SourcedMetric:
        return dict(sorted(sources.items(), key=cls._len_source))

    @staticmethod
    def _filter_by_epoch(
        sourced_array: SourcedArray, start: int
    ) -> SourcedArray:
        if start == 1:
            return sourced_array

        filtered: SourcedArray = {}
        for name, data in sourced_array.items():
            mask = data[:, 0] >= start  # the epoch is in column 0
            if np.any(mask):
                filtered[name] = data[mask]

        return filtered

    @staticmethod
    def _len_source(source_pair: tuple[str, HistoryMetric]) -> int:
        return -len(source_pair[1][0])  # does not reverse when equal

    @staticmethod
    def _source_to_numpy(sourced_metric: SourcedMetric) -> SourcedArray:
        return {
            name: np.column_stack((epochs, values))
            for name, (epochs, values) in sourced_metric.items()
        }

"""Abstract classes for trackers."""

from __future__ import annotations

import abc
import functools
import math
import pathlib

from collections.abc import Generator, Sequence
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
    'TestResults',
]


HistoryMetric: TypeAlias = tuple[list[int], list[float]]
HistoryMetrics: TypeAlias = tuple[list[int], dict[str, list[float]]]
SourcedMetric: TypeAlias = dict[str, HistoryMetric]
SourcedMetrics: TypeAlias = dict[str, HistoryMetrics]
NpArray: TypeAlias = npt.NDArray[np.float64]
SourcedArray: TypeAlias = dict[str, NpArray]
TestResults: TypeAlias = dict[str, list[tuple[int, dict[str, float]]]]

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
    _record: bool
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
        self._record = True
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
    def record(self) -> bool:
        """Whether the current run is recorded, so that the tracker writes."""
        return self._record

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
        super().notify(event)
        self._par_dir = event.par_dir
        self._exp_name = event.exp_name
        self._record = event.record
        self._run_id = event.run_id
        if self.record:
            self.par_dir.mkdir(exist_ok=True, parents=True)

        return

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
        self._record = event.record
        return super().notify(event)

    @override
    def clean_up(self) -> None:
        """Remove experimental data from the tracker."""
        self._par_dir = None
        self._exp_name = None
        self._record = True
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
            NoSourcesError: if the model has no sources.
            ValueError: if max_epoch is less than -1.
        """
        if max_epoch == 0:
            return {}

        if max_epoch < -1:
            raise ValueError('Max epoch should not be less than -1.')

        metrics = self._load_metrics(model_name, max_epoch)
        if not metrics:
            raise exceptions.NoSourcesError(self, model_name)

        return metrics

    @abc.abstractmethod
    def _load_metrics(
        self, model_name: str, max_epoch: int = -1
    ) -> SourcedMetrics: ...

    def load_test_metrics(self, model_name: str) -> SourcedMetrics:
        """Load test metrics from the last run of the experiment.

        Args:
            model_name: the name of the model.

        Returns:
            The epochs and named metric values of each test source.
        """
        return self._load_test_metrics(model_name)

    @abc.abstractmethod
    def _load_test_metrics(self, model_name: str) -> SourcedMetrics: ...


class MemoryMetrics(tracking.Tracker):
    """Keep all metrics in memory.

    Attributes:
        model_dict: training and validation metrics recorded in this session.
        test_dict: test metrics recorded in this session.
    """

    model_dict: Final[dict[str, SourcedMetrics]]
    test_dict: Final[dict[str, SourcedMetrics]]
    _metric_loader: Final[MetricLoader | None]

    def __init__(self, *, metric_loader: MetricLoader | None = None) -> None:
        """Initialize.

        Args:
            metric_loader: object to load the metrics.
        """
        super().__init__()
        self._metric_loader = metric_loader
        self.model_dict = {}
        self.test_dict = {}
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.LoadModelEvent) -> None:
        if self._metric_loader is None:
            return super().notify(event)

        self.model_dict[event.model_name] = self._load_metrics(
            event.model_name, event.epoch
        )
        test_metrics = self._metric_loader.load_test_metrics(event.model_name)
        self.test_dict[event.model_name] = test_metrics
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        self._record_metric(self.model_dict, event)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.TestMetricEvent) -> None:
        self._record_metric(self.test_dict, event)
        return super().notify(event)

    def _load_metrics(
        self, model_name: str, max_epoch: int = -1
    ) -> SourcedMetrics:
        if self._metric_loader is None:
            return {}

        try:
            return self._metric_loader.load_metrics(model_name, max_epoch)
        except exceptions.NoSourcesError:
            return {}

    @staticmethod
    def _record_metric(
        target_dict: dict[str, SourcedMetrics],
        event: log_events.MetricEvent,
    ) -> None:
        sourced_metrics = target_dict.setdefault(event.model_name, {})
        epochs, logs_dict = sourced_metrics.setdefault(
            event.source_name, ([], {})
        )
        epochs.append(event.epoch)
        for metric_name, metric_value in event.metrics.items():
            logs_dict.setdefault(metric_name, []).append(metric_value)

        return


class BasePlotter(MemoryMetrics, Dumper, abc.ABC, Generic[Figure]):
    """Abstract class for plotting the learning curves on request.

    Attributes:
        file_suffix: the suffix of the files the figures are saved to.
        show: whether to render the plots when saved.
    """

    file_suffix: ClassVar[str]
    show: Final[bool]

    def __init__(
        self,
        *,
        metric_loader: MetricLoader | None = None,
        par_dir: pathlib.Path | None = None,
        show: bool = False,
    ) -> None:
        """Initialize.

        Args:
            metric_loader: a tracker that can load metrics from a previous run.
            par_dir: the parent directory for the saved figures. Default uses
                the same of the current experiment.
            show: whether to render the plots when saved.
        """
        super().__init__(metric_loader=metric_loader)
        self.user_par_dir = par_dir
        self.show = show
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.TerminatedTrainingEvent) -> None:
        self._save_training_plots(event.model_name)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndTrainingEvent) -> None:
        self._save_training_plots(event.model_name)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.TestMetricEvent) -> None:
        super().notify(event)
        self._save_test_plot(event.model_name)
        return

    def create_training_plots(
        self,
        model_name: str,
        *,
        metric_names: Sequence[str] = (),
        source_names: Sequence[str] = (),
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
            ValueError: if no metric is left to plot.
        """
        plots = self._create_training_plots(
            model_name, metric_names, source_names, start_epoch
        )
        if not plots:
            msg = f'No metric to plot for model {model_name}.'
            raise ValueError(msg)

        return plots

    def create_test_plot(
        self,
        model_name: str,
        *,
        metric_names: Sequence[str] = (),
        source_names: Sequence[str] = (),
    ) -> Figure | None:
        """Create the figure of the tests of a model without rendering it.

        Args:
            model_name: the name of the model to plot.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the tests to plot. Defaults to all.

        Returns:
            A figure of the tests, or None if the model has no test left to
            plot.
        """
        test_results = self._get_test_results(
            model_name, metric_names, source_names
        )
        if not test_results:
            return None

        return self._create_test_plot(model_name, test_results)

    def plot(
        self,
        model_name: str,
        *,
        metric_names: Sequence[str] = (),
        source_names: Sequence[str] = (),
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
            ValueError: if neither a metric nor a test is left to plot.
        """
        training_plots, test_plot = self._create_plots(
            model_name, metric_names, source_names, start_epoch
        )
        plots = list(training_plots.values())
        if test_plot is not None:
            plots.append(test_plot)

        self._render(plots)
        return

    def save_plots(
        self,
        model_name: str,
        *,
        metric_names: Sequence[str] = (),
        source_names: Sequence[str] = (),
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
            ValueError: if neither a metric nor a test is left to plot.

        Side Effects:
            Writes one file for each metric, and one for the tests, in the
            model folder, if the run is recorded.
        """
        training_plots = self._save_training_plots(
            model_name, metric_names, source_names, start_epoch
        )
        test_plot = self._save_test_plot(model_name, metric_names, source_names)
        if not training_plots and test_plot is None:
            msg = f'No metric to plot for model {model_name}.'
            raise ValueError(msg)

        return

    def _get_model_dir(self, model_name: str) -> pathlib.Path:
        model_dir = self._get_run_dir() / model_name
        model_dir.mkdir(exist_ok=True)
        return model_dir

    def _save_training_plots(
        self,
        model_name: str,
        metric_names: Sequence[str] = (),
        source_names: Sequence[str] = (),
        start_epoch: int = 1,
    ) -> dict[str, Figure]:
        plots = self._create_training_plots(
            model_name, metric_names, source_names, start_epoch
        )
        if self.record:
            model_dir = self._get_model_dir(model_name)
            for metric_name, plot in plots.items():
                self._save_plot(
                    plot, model_dir / f'{metric_name}{self.file_suffix}'
                )

        if self.show and plots:
            self._render(list(plots.values()))

        return plots

    def _save_test_plot(
        self,
        model_name: str,
        metric_names: Sequence[str] = (),
        source_names: Sequence[str] = (),
    ) -> Figure | None:
        plot = self.create_test_plot(
            model_name, metric_names=metric_names, source_names=source_names
        )
        if plot is None:
            return None

        if self.record:
            self._save_plot(
                plot,
                self._get_model_dir(model_name)
                / f'test_metrics{self.file_suffix}',
            )

        if self.show:
            self._render([plot])

        return plot

    def _create_plots(
        self,
        model_name: str,
        metric_names: Sequence[str],
        source_names: Sequence[str],
        start_epoch: int,
    ) -> tuple[dict[str, Figure], Figure | None]:
        training_plots = self._create_training_plots(
            model_name, metric_names, source_names, start_epoch
        )
        test_plot = self.create_test_plot(
            model_name, metric_names=metric_names, source_names=source_names
        )
        if not training_plots and test_plot is None:
            msg = f'No metric to plot for model {model_name}.'
            raise ValueError(msg)

        return training_plots, test_plot

    def _create_training_plots(
        self,
        model_name: str,
        metric_names: Sequence[str],
        source_names: Sequence[str],
        start_epoch: int,
    ) -> dict[str, Figure]:
        metric_arrays = self._get_metric_arrays(
            model_name, metric_names, source_names, start_epoch
        )
        return {
            metric_name: self._create_training_plot(
                model_name, metric_name, sourced_array
            )
            for metric_name, sourced_array in metric_arrays.items()
        }

    @abc.abstractmethod
    def _create_training_plot(
        self, model_name: str, metric_name: str, sourced_array: SourcedArray
    ) -> Figure:
        """Create the figure of a metric from its sourced arrays."""

    @abc.abstractmethod
    def _create_test_plot(
        self, model_name: str, test_results: TestResults
    ) -> Figure:
        """Create the figure of the tests of a model, one panel per metric."""

    def _get_metric_arrays(
        self,
        model_name: str,
        metric_names: Sequence[str],
        source_names: Sequence[str],
        start_epoch: int,
    ) -> dict[str, SourcedArray]:
        if start_epoch < 1:
            raise ValueError('Start epoch must be positive.')

        self._load_missing_metrics(model_name)
        sourced_metrics = self.model_dict.get(model_name, {})

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

        return metric_arrays

    def _load_missing_metrics(self, model_name: str) -> None:
        if self._metric_loader is None:
            return

        if model_name not in self.model_dict:
            self.model_dict[model_name] = self._load_metrics(model_name)

        if model_name not in self.test_dict:
            self.test_dict[model_name] = self._metric_loader.load_test_metrics(
                model_name
            )

        return

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

    def _get_test_results(
        self,
        model_name: str,
        metric_names: Sequence[str],
        source_names: Sequence[str],
    ) -> TestResults:
        self._load_missing_metrics(model_name)
        test_metrics = self.test_dict.get(model_name, {})
        out: TestResults = {}
        for source_name in sorted(test_metrics):
            if source_names and source_name not in source_names:
                continue

            epochs, metrics_dict = test_metrics[source_name]
            if not epochs:
                continue

            evaluations = self._extract_evaluations(
                self, source_name, epochs, metrics_dict, metric_names
            )
            if evaluations:
                out[source_name] = evaluations

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
    def _save_plot(self, figure: Figure, path: pathlib.Path) -> None:
        """Save the figure to the path."""

    @classmethod
    def _order_sources(cls, sources: SourcedMetric) -> SourcedMetric:
        return dict(sorted(sources.items(), key=cls._len_source))

    @staticmethod
    def _extract_evaluations(
        tracker: tracking.Tracker,
        source_name: str,
        epochs: list[int],
        metrics_dict: dict[str, list[float]],
        metric_names: Sequence[str],
    ) -> list[tuple[int, dict[str, float]]]:
        for name, values in metrics_dict.items():
            if metric_names and name not in metric_names:
                continue

            if len(epochs) != len(values):
                msg = '{} and {} logs refer to different epochs.'
                msg = msg.format(source_name, name)
                raise exceptions.TrackerError(tracker, msg)

        evaluations: dict[int, dict[str, float]] = {}
        for i, epoch in enumerate(epochs):
            metrics: dict[str, float] = {}
            for name, values in metrics_dict.items():
                if metric_names and name not in metric_names:
                    continue

                metrics[name] = values[i]

            if metrics:
                evaluations[epoch] = metrics

        return sorted(evaluations.items())

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
    def _get_grid_shape(n_panels: int) -> tuple[int, int]:
        n_rows = math.ceil(math.sqrt(n_panels))
        return n_rows, math.ceil(n_panels / n_rows)

    @staticmethod
    def _get_test_metric_names(test_results: TestResults) -> list[str]:
        names = (
            set(metrics)
            for entries in test_results.values()
            for _, metrics in entries
        )
        return sorted(set().union(*names))

    @staticmethod
    def _len_source(source_pair: tuple[str, HistoryMetric]) -> int:
        return -len(source_pair[1][0])  # does not reverse when equal

    @staticmethod
    def _source_to_numpy(sourced_metric: SourcedMetric) -> SourcedArray:
        return {
            name: np.column_stack((epochs, values))
            for name, (epochs, values) in sourced_metric.items()
        }

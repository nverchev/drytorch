"""Module containing abstract classes for trackers."""

import abc
import functools
import pathlib
from typing import Generic, Iterable, Optional, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from drytorch import log_events
from drytorch import exceptions
from drytorch import experiments
from drytorch import tracking

HistoryMetric: TypeAlias = tuple[list[int], list[float]]
HistoryMetrics: TypeAlias = tuple[list[int], dict[str, list[float]]]
SourcedMetric: TypeAlias = dict[str, HistoryMetric]
SourcedMetrics: TypeAlias = dict[str, HistoryMetrics]
NpArray: TypeAlias = npt.NDArray[np.float64]
SourcedArray: TypeAlias = dict[str, NpArray]

Plot = TypeVar('Plot')


class Dumper(tracking.Tracker):
    """Dump metrics or metadata in a custom directory."""

    def __init__(self, par_dir: Optional[pathlib.Path] = None):
        """
        Args:
            par_dir: the directory where to dump metadata. Default uses the one
                  for the current experiment.
        """
        super().__init__()
        self._par_dir = par_dir
        self._exp_dir: Optional[pathlib.Path] = None

    @property
    def par_dir(self) -> pathlib.Path:
        """Return the directory where the files will be saved."""
        if self._par_dir is None:
            if self._exp_dir is None:
                raise exceptions.AccessOutsideScopeError()
            path = self._exp_dir
        else:
            path = self._par_dir

        path.mkdir(exist_ok=True)
        return path

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        self._exp_dir = event.exp_dir

    @notify.register
    def _(self, _: log_events.StopExperiment) -> None:
        self._exp_dir = None


class MetricLoader(tracking.Tracker, abc.ABC):
    """Interface for trackers that load metrics."""

    def load_metrics(self,
                     model_name: str,
                     max_epoch: int = -1) -> SourcedMetrics:
        """
        Load metrics stored by the tracker.

        Args:
            model_name: the name of the model.
            max_epoch: the maximum epoch to load. Defaults to all.

        Returns:
            The current epochs and named metric values by the source.

        Raises:
            AccessOutsideScopeError: if called outside the experiment scope.
        """
        try:
            experiments.Experiment.current()
        except exceptions.NoActiveExperimentError:
            raise exceptions.AccessOutsideScopeError()

        if max_epoch == 0:
            return {}

        if max_epoch < -1:
            raise ValueError('Max epoch should not be less than -1.')

        return self._load_metrics(model_name, max_epoch)

    @abc.abstractmethod
    def _load_metrics(self,
                      model_name: str,
                      max_epoch: int = -1) -> SourcedMetrics:
        ...


class MemoryMetrics(tracking.Tracker):
    """
    Keep all metrics in memory.

    Attributes:
        model_dict: all metrics recorded in this session.
    """

    def __init__(self, metric_loader: Optional[MetricLoader] = None) -> None:
        super().__init__()
        self._metric_loader = metric_loader
        self.model_dict = dict[str, SourcedMetrics]()

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        sourced_metrics = self.model_dict.setdefault(event.model_name, {})
        epochs, logs_dict = sourced_metrics.setdefault(event.source_name,
                                                       ([], {}))
        epochs.append(event.epoch)
        for metric_name, metric_value in event.metrics.items():
            logs_dict.setdefault(metric_name, []).append(metric_value)

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.LoadModel) -> None:
        if self._metric_loader is None:
            return None

        metrics = self._metric_loader.load_metrics(event.model_name,
                                                   event.epoch)
        self.model_dict[event.model_name] = metrics
        return super().notify(event)


class BasePlotter(MemoryMetrics, Generic[Plot]):
    """Abstract class for plotting trajectory from sources. """

    def __init__(self,
                 model_names: Iterable[str] = (),
                 source_names: Iterable[str] = (),
                 metric_names: Iterable[str] = (),
                 start: int = 1,
                 metric_loader: Optional[MetricLoader] = None) -> None:
        """
        Args:
            model_names: the names of the models to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            metric_names: the names of the metrics to plot. Defaults to all.
            start: if positive, the epoch from which to start plotting;
                if negative, the last number of epochs. Defaults to all.
            metric_loader: a tracker that can load metrics from a previous run.
        Note:
            start_epoch allows you to exclude the initial epochs from the graph.
            During the first 2 * start_epoch epochs, the graph is shown in
            its entirety.
        """
        super().__init__(metric_loader)
        self._model_names = model_names
        self._source_names = source_names
        self._metric_names = metric_names
        self._start = start
        self._removed_start = False

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndEpoch) -> None:
        super().notify(event)
        if self._start < 0:
            start = max(1, event.epoch + self._start)
        else:
            start = self._start if event.epoch >= 2 * self._start else 1

        self._update_plot(model_name=event.model_name,
                          start=start)

    @notify.register
    def _(self, event: log_events.EndTest) -> None:
        super().notify(event)
        start = max(1, self._start)
        self._update_plot(model_name=event.model_name, start=start)
        return

    def plot(self,
             model_name: str,
             source_names: Iterable[str] = (),
             metric_names: Iterable[str] = (),
             start_epoch: int = 1) -> list[Plot]:
        """
        Plot the learning curves.

        Args:
            model_name: the name of the model to plot.
            source_names: the names of the sources to plot. Defaults to all.
            metric_names: the metric to plot. Defaults to all.
            start_epoch: the epoch from which to start plotting.

        Returns:
            References to the plot objects or windows depending on the backend.
        """
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

        return self._plot(model_name, source_names, metric_names, start_epoch)

    def _plot(self,
              model_name: str,
              source_names: Iterable[str],
              metric_names: Iterable[str],
              start: int) -> list[Plot]:

        sourced_metrics = self.model_dict.get(model_name, {})
        if source_names:
            sourced_metrics = {source: sourced_metrics[source]
                               for source in source_names
                               if source in sourced_metrics}
        if not metric_names:
            all_metrics = (set(logs[1]) for logs in sourced_metrics.values())
            metric_names = sorted(set().union(*all_metrics))

        plots = list[Plot]()
        self._prepare_layout(model_name, list(metric_names))
        for metric_name in metric_names:
            processed_sources = self._process_source(sourced_metrics,
                                                     metric_name,
                                                     start)
            if processed_sources:
                plots.append(self._plot_metric(model_name,
                                               metric_name,
                                               **processed_sources))

        return plots

    @abc.abstractmethod
    def _plot_metric(self,
                     model_name: str,
                     metric_name: str,
                     **sourced_array: NpArray) -> Plot:
        ...

    def _prepare_layout(self, model_name: str, metric_names: list[str]) -> None:
        return

    def _process_source(self,
                        sourced_metrics: SourcedMetrics,
                        metric_name: str,
                        start: int) -> SourcedArray:
        sourced_metric = self._filter_metric(sourced_metrics, metric_name)
        ordered_sources = self._order_sources(sourced_metric)
        sourced_array = self._source_to_numpy(ordered_sources)
        return self._filter_by_epoch(sourced_array, start)

    def _update_plot(self, model_name: str, start: int) -> None:
        if self._model_names and model_name not in self._model_names:
            return

        self._plot(model_name, self._source_names, self._metric_names, start)
        return

    @classmethod
    def _order_sources(cls, sources: SourcedMetric) -> SourcedMetric:
        return dict(sorted(sources.items(), key=cls._len_source))

    @staticmethod
    def _filter_metric(sourced_metrics: SourcedMetrics,
                       metric_name: str) -> SourcedMetric:
        return {source_name: (epochs, metrics[metric_name])
                for source_name, (epochs, metrics) in sourced_metrics.items()
                if epochs and metric_name in metrics}

    @staticmethod
    def _filter_by_epoch(sourced_array: SourcedArray,
                         start: int) -> SourcedArray:
        if start == 1:
            return sourced_array

        filtered = {}
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
        return {name: np.column_stack((epochs, values))
                for name, (epochs, values) in sourced_metric.items()}

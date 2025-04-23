"""Abstract classes for trackers."""

import abc
import functools
import pathlib
from typing import Optional, Iterable

from typing_extensions import override

from dry_torch import exceptions
from dry_torch import experiments
from dry_torch import log_events
from dry_torch import tracking

LogTuple = tuple[list[int], dict[str, list[float]]]


class AbstractDumper(tracking.Tracker, metaclass=abc.ABCMeta):
    """Tracker that for a custom directory."""

    def __init__(self, par_dir: Optional[pathlib.Path] = None):
        """
        Args:
            par_dir: Directory where to dump metadata. Defaults uses the one of
                the current experiment.
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
                     max_epoch: int = -1) -> dict[str, LogTuple]:
        """
        Loads metrics stored by the tracker.

        Args:
            model_name: name of the model.
            max_epoch: maximum epoch to load. Defaults to all.

        Returns:
            current epochs and named metric values by source.

        Raises:
            AccessOutsideScopeError if called outside the experiment scope.
        """
        try:
            experiments.Experiment.current()
        except exceptions.NoActiveExperimentError:
            raise exceptions.AccessOutsideScopeError()

        if max_epoch == 0:
            return {}
        if max_epoch < -1:
            msg = 'Max epoch should not be less than -1.'
            raise exceptions.TrackerException(self, msg)
        return self._load_metrics(model_name, max_epoch)

    @abc.abstractmethod
    def _load_metrics(self,
                      model_name: str,
                      max_epoch: int = -1) -> dict[str, LogTuple]:
        ...


class MemoryMetrics(tracking.Tracker):
    """Keeps all metrics in memory."""

    def __init__(self, metric_loader: Optional[MetricLoader] = None) -> None:
        super().__init__()
        self.metric_loader = metric_loader
        self.model_metrics = dict[str, dict[str, LogTuple]]()

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        source_dict = self.model_metrics.setdefault(event.model_name, {})
        epochs, logs_dict = source_dict.setdefault(event.source_name, ([], {}))
        epochs.append(event.epoch)
        for metric_name, metric_value in event.metrics.items():
            logs_dict.setdefault(metric_name, []).append(metric_value)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.LoadModel) -> None:
        if self.metric_loader is None:
            return
        metrics = self.metric_loader.load_metrics(event.model_name, event.epoch)
        self.model_metrics[event.model_name] = metrics
        return super().notify(event)


class BasePlotter(MemoryMetrics):
    """Abstract class for plotting trajectory from sources. """

    def __init__(self,
                 model_names: Iterable[str] = (),
                 metric_names: Iterable[str] = (),
                 metric_loader: Optional[MetricLoader] = None,
                 start_epoch: int = 1) -> None:
        """
        Args:
            model_names: the names of the models to plot. Defaults to all.
            metric_names: the names of the metrics to plot. Defaults to all.
            metric_loader: a tracker that can load metrics from a previous run.
            start_epoch: epoch from which plot starts.

        Note:
            start_epoch allows you to exclude the initial epochs from the graph.
            During the first 2 * start_epoch epochs, the graph is shown in
            its entirety.
        """
        super().__init__(metric_loader)
        self._model_names = model_names
        self._metric_names = metric_names
        self._start = start_epoch
        self._removed_start = False

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.EndEpoch) -> None:
        super().notify(event)
        if self._model_names and event.model_name not in self._model_names:
            return
        source_dict = self.model_metrics.get(event.model_name, {})
        if event.epoch >= 2 * self._start and self._removed_start:
            source_dict = self._remove_start(source_dict, self._start)
            self._removed_start = True

        if source_dict:
            if self._metric_names:
                metric_names: Iterable[str] = self._metric_names
            else:
                all_metrics = (list(logs[1]) for logs in source_dict.values())
                metric_names = sum(all_metrics, [])

            for metric_name in metric_names:
                filtered_sources = self._filter_metric(source_dict, metric_name)
                self._plot_metric(event.model_name,
                                  metric_name,
                                  **filtered_sources)
        return

    def plot(self,
             model_name: str,
             source_names: Iterable[str] = (),
             metric_names: Iterable[str] = (),
             start: int = 1) -> None:
        """
        Plots the learning curves.

        Args:
            model_name: the name of the model to plot.
            source_names: the names of the sources to plot. Defaults to all.
            metric_names: the metric to plot. Defaults to all.
            start: the epoch from which to start plotting. Defaults to 1.
        """
        source_dict = self.model_metrics.get(model_name, {})
        if not source_dict and self.metric_loader is not None:
            source_dict = self.metric_loader.load_metrics(model_name)
        if not source_dict:
            msg = f'No model named {model_name} has been found.'
            raise exceptions.TrackerException(self, msg)
        if source_names:
            source_dict = {source: logs
                           for source, logs in source_dict.items()
                           if source in source_names}
        source_dict = self._remove_start(source_dict, start)
        if metric_names:
            metric_names = metric_names
        else:
            all_metrics = (list(logs[1]) for logs in source_dict.values())
            metric_names = sum(all_metrics, [])

        for metric_name in metric_names:
            filtered_sources = self._filter_metric(source_dict, metric_name)
            self._plot_metric(model_name, metric_name, **filtered_sources)

    @abc.abstractmethod
    def _plot_metric(self,
                     model_name: str,
                     metric_name: str,
                     **sources: tuple[list[int], list[float]]) -> None:
        ...

    @staticmethod
    def _remove_start(source_dict: dict[str, LogTuple],
                      start: int) -> dict[str, LogTuple]:

        out = dict[str, LogTuple]()

        for source_name, (epochs, metric_dict) in source_dict.items():
            new_epochs = list[int]()
            new_metric_dict = dict[str, list[float]]()
            for metric_name in metric_dict:
                for epoch, value in zip(epochs, metric_dict[metric_name]):
                    if epoch >= start:
                        new_epochs.append(epoch)
                        value_list = new_metric_dict.setdefault(metric_name, [])
                        value_list.append(value)
            out[source_name] = (new_epochs, new_metric_dict)
        return out

    @staticmethod
    def _filter_metric(
            source_dict: dict[str, LogTuple],
            metric_name: str,
    ) -> dict[str, tuple[list[int], list[float]]]:
        return {source: (epochs, metrics[metric_name])
                for source, (epochs, metrics) in source_dict.items() if epochs}

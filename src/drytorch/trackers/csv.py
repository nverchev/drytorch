"""Module containing a tracker that dumps metrics in a CSV file."""

import csv
import functools
import pathlib

from typing import ClassVar, Final

from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.trackers import base_classes


class DryTorchDialect(csv.Dialect):
    """Dialect similar to excel that converts numbers to floats."""

    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = csv.QUOTE_NONNUMERIC


class CSVDumper(base_classes.Dumper, base_classes.MetricLoader):
    """Dump metrics into a CSV file."""

    folder_name = 'csv_metrics'
    _default_dialect: ClassVar[csv.Dialect] = DryTorchDialect()
    _base_headers: ClassVar[tuple[str, ...]] = ('Model', 'Source', 'Epoch')

    def __init__(
        self,
        par_dir: pathlib.Path | None = None,
        dialect: csv.Dialect = _default_dialect,
    ) -> None:
        """Constructor.

        Args:
            par_dir: the parent directory for the tracker data. Default uses
                the same of the current experiment.
            dialect: the format specification. Defaults to local dialect.
        """
        super().__init__(par_dir)
        self._active_sources: Final = set[str]()
        self._dialect: csv.Dialect = dialect
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        self._resume_run = event.resumed
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        run_dir = self._get_run_dir()
        file_address = self._file_path(
            run_dir, event.model_name, event.source_name
        )
        metric_names = tuple(event.metrics)
        headers = self._base_headers + metric_names
        if event.source_name not in self._active_sources:
            if self._resume_run and file_address.exists():
                with file_address.open(newline='') as log:
                    reader = csv.reader(log, dialect=self._dialect)
                    previous_headers = tuple(next(reader))
                    if headers != previous_headers:
                        msg = (
                            f'Current {headers=} and previous headers='
                            f'{previous_headers} do not correspond'
                        )
                        raise exceptions.TrackerError(self, msg)

            else:
                with file_address.open('w', newline='') as log:  # reset
                    writer = csv.writer(log, dialect=self._dialect)
                    writer.writerow(headers)

            self._active_sources.add(event.source_name)
        with file_address.open('a', newline='') as log:
            writer = csv.writer(log, dialect=self._dialect)
            writer.writerow(
                [
                    event.model_name,
                    event.source_name,
                    event.epoch,
                    *event.metrics.values(),
                ]
            )
        return super().notify(event)

    def read_csv(
        self,
        model_name: str,
        source: str,
        max_epoch: int = -1,
    ) -> base_classes.HistoryMetrics:
        """Read the CSV file associated with the given model and source.

        Args:
            model_name: the name of the model.
            source: the source of the metrics.
            max_epoch: the maximum number of epochs to load. Defaults to all.

        Returns:
            Epochs and relative value for each metric.
        """
        run_dir = self._get_run_dir()
        file_address = self._file_path(run_dir, model_name, source)

        with file_address.open(newline='') as log:
            reader = csv.reader(log, dialect=self._dialect)
            headers = next(reader)
            len_base = len(self._base_headers)
            metric_names = headers[len_base:]
            epochs = list[int]()
            named_metric_values = dict[str, list[float]]()
            epoch_column = self._base_headers.index('Epoch')
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                epoch = int(row[epoch_column])
                if epochs and epochs[-1] >= epoch:  # only load the last run
                    epochs.clear()
                    named_metric_values.clear()

                if max_epoch != -1 and epoch > max_epoch:
                    continue

                epochs.append(epoch)
                for metric, value in zip(
                    metric_names, row[len_base:], strict=True
                ):
                    value_list = named_metric_values.setdefault(metric, [])
                    value_list.append(float(value))

            return epochs, named_metric_values

    @override
    def clean_up(self) -> None:
        self._resume_run = False
        return super().clean_up()

    def _find_sources(self, model_name: str) -> set[str]:
        path = self._get_run_dir() / model_name
        named_sources = {file.stem for file in path.glob('*.csv')}
        if not named_sources:
            msg = f'No sources in {path}.'
            raise exceptions.TrackerError(self, msg)

        return named_sources

    @staticmethod
    def _file_path(
        run_dir: pathlib.Path,
        model_name: str,
        source_name: str,
    ) -> pathlib.Path:
        model_path = run_dir / model_name
        model_path.mkdir(exist_ok=True)
        return model_path / f'{source_name}.csv'

    def _load_metrics(
        self, model_name: str, max_epoch: int = -1
    ) -> base_classes.SourcedMetrics:
        out: base_classes.SourcedMetrics = {}

        if self._resume_run:
            sources = self._find_sources(model_name)
        elif self._active_sources:
            sources = self._active_sources
        else:
            sources = set()

        for source in sources:
            out[source] = self.read_csv(model_name, source, max_epoch)

        return out

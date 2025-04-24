"""Tracker that dumps metrics in a CSV file."""

import csv
import functools
import pathlib
from typing import Optional

from typing_extensions import override

from dry_torch import exceptions
from dry_torch import log_events
from dry_torch.trackers import base_classes


class DryTorchDialect(csv.Dialect):
    """Dialect similar to excel that converts numbers to floats."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = csv.QUOTE_NONNUMERIC


class CSVDumper(base_classes.AbstractDumper,
                base_classes.MetricLoader):
    """
    Tracker that dumps metrics into a CSV file.

    Attributes:
        resume_run: resume previous run from the project."""

    def __init__(self,
                 par_dir: Optional[pathlib.Path] = None,
                 dialect: csv.Dialect = DryTorchDialect(),
                 resume_run: bool = False) -> None:
        """
        Args:
            par_dir: directory where to dump metadata. Defaults to the one for
                the current experiment.
            dialect: class with format specification. Defaults to local dialect.
        """
        super().__init__(par_dir)
        self.resume_run = resume_run
        self._active_sources = set[str]()
        self._dialect = dialect
        self._exp_dir: Optional[pathlib.Path] = None
        self._base_headers = ('Model', 'Source', 'Epoch')

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        file_address = self.file_name(event.model_name, event.source_name)
        metric_names = tuple(event.metrics)
        headers = self._base_headers + metric_names
        if event.source_name not in self._active_sources:
            if self.resume_run:
                with file_address.open() as log:
                    reader = csv.reader(log, dialect=self._dialect)
                    previous_headers = tuple(next(reader))
                    if headers != previous_headers:
                        msg = (f'Current {headers=} and previous headers='
                               f'{previous_headers} do not correspond')
                        raise exceptions.TrackerException(self, msg)
            else:
                with file_address.open('w') as log:  # reset the file.
                    writer = csv.writer(log, dialect=self._dialect)
                    writer.writerow(headers)
            self._active_sources.add(event.source_name)
        with file_address.open('a') as log:
            writer = csv.writer(log, dialect=self._dialect)
            writer.writerow([event.model_name,
                             event.source_name,
                             event.epoch,
                             *event.metrics.values()])
        return super().notify(event)

    def _csv_path(self, model_name: str) -> pathlib.Path:
        path = self.par_dir / model_name / 'csv_metrics'
        path.mkdir(exist_ok=True, parents=True)
        return path

    def file_name(self, model_name: str, source_name: str) -> pathlib.Path:
        """
        Return path to the csv file.

        Args:
            model_name: name of the model.
            source_name: source of the metrics.
        Returns:
            path to the csv file.
        """
        path = self._csv_path(model_name)
        return (path / source_name).with_suffix('.csv')

    def _find_sources(self, model_name: str) -> set[str]:
        path = self._csv_path(model_name)
        named_sources = {file.stem for file in path.glob('*.csv')}
        if not named_sources:
            msg = f'No sources for model {model_name}.'
            raise exceptions.TrackerException(self, msg)
        return named_sources

    def read_csv(self,
                 model_name: str,
                 source: str,
                 max_epoch: Optional[int] = None,
                 ) -> tuple[list[int], dict[str, list[float]]]:
        """
        Reads the CSV file associated with the given model and source.

        Args:
            model_name: name of the model.
            source: source of the metrics.
            max_epoch: maximum number of epochs to load. Defaults to all.
        Returns:
            column headers and the data as list of list (of float when using
                the default dialect).
        """
        file_address = self.file_name(model_name, source)
        with file_address.open() as log:
            reader = csv.reader(log, dialect=self._dialect)
            headers = next(reader)
            len_base = len(self._base_headers)
            metric_names = headers[len_base:]
            epochs = list[int]()
            named_metric_values = dict[str, list[float]]()
            epoch_column = self._base_headers.index('Epoch')
            for row in reader:
                epoch = int(row[epoch_column])
                if epochs and epochs[-1] >= epoch:  # only load last run
                    epochs.clear()
                    named_metric_values.clear()
                if max_epoch != -1 and epoch > max_epoch:
                    continue
                epochs.append(epoch)
                for metric, value in zip(metric_names, row[len_base:]):
                    value_list = named_metric_values.setdefault(metric, [])
                    value_list.append(float(value))
            return epochs, named_metric_values

    def _load_metrics(self,
                      model_name: str,
                      max_epoch: int = -1) -> dict[str, base_classes.LogTuple]:

        out = dict[str, tuple[list[int], dict[str, list[float]]]]()
        if self._active_sources:
            sources = self._active_sources
        elif self.resume_run:
            sources = self._find_sources(model_name)
        else:
            sources = set()
        for source in sources:
            out[source] = self.read_csv(model_name, source, max_epoch)
        return out

"""Tracker that dumps metrics in a CSV file."""

import csv
import functools
import pathlib
from typing import Optional

from typing_extensions import override

from dry_torch import log_events
from dry_torch.trackers import abstract_dumper
from dry_torch.trackers import plotting


class DryTorchDialect(csv.Dialect):
    """Dialect similar to excel that converts numbers to floats."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = csv.QUOTE_NONNUMERIC


class CSVDumper(abstract_dumper.AbstractDumper, plotting.MetricLoader):
    """Tracker that dumps metrics into a CSV file."""

    def __init__(self,
                 par_dir: Optional[pathlib.Path] = None,
                 dialect: csv.Dialect = DryTorchDialect()) -> None:
        """
        Args:
            par_dir: directory where to dump metadata. Defaults to the one for
                the current experiment.
            dialect: class with format specification. Defaults to local dialect.
        """
        super().__init__(par_dir)
        self._dialect = dialect
        self._exp_dir: Optional[pathlib.Path] = None

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        model_name = format(event.model_name, 's')
        source_name = format(event.source, 's')
        file_address = self.file_name(model_name, source_name)
        with file_address.open('a') as log:
            writer = csv.writer(log, dialect=self._dialect)
            if not log.tell():
                writer.writerow(['Model', 'Source', 'Epoch', *event.metrics])
            writer.writerow([event.model_name,
                             event.source,
                             event.epoch,
                             *event.metrics.values()])
        return

    def csv_path(self, model_name: str) -> pathlib.Path:
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
        path = self.csv_path(model_name)
        return (path / source_name).with_suffix('.csv')

    def find_source(self, model_name: str) -> list[str]:
        path = self.csv_path(model_name)
        return [file.name for file in path.glob('*.csv')]

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
            columns = next(reader)
            metric_names = columns[3:]
            epochs = list[int]()
            metric_values = dict[str, list[float]]()
            for row in reader:
                epoch = int(row[2])
                if max_epoch is not None and epoch > max_epoch:
                    break
                epochs.append(epoch)
                for metric, value in zip(metric_names, row[3:]):
                    metric_values.setdefault(metric, []).append(float(value))
            return epochs, metric_values

    def load_metrics(
            self,
            model_name: str,
            max_epoch: Optional[int] = None,
    ) -> dict[str, tuple[list[int], dict[str, list[float]]]]:
        """
        Loads metrics from the CSV file.

        Args:
            model_name: name of the model.
            max_epoch: maximum number of epochs to load. Defaults to all.
        """
        model_name = format(model_name, 's')
        sources = self.find_source(model_name)
        out = dict[str, tuple[list[int], dict[str, list[float]]]]()
        for source in sources:
            out[source] = self.read_csv(model_name, source, max_epoch)
        return out

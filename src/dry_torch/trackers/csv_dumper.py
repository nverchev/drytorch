"""Tracker that dumps metrics in a CSV file."""

import csv
import functools
import pathlib
from typing import Any, Optional

from typing_extensions import override

from dry_torch import log_events
from dry_torch.trackers import abstract_dumper


class DryTorchDialect(csv.Dialect):
    """Dialect similar to excel that convert number to float."""
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = csv.QUOTE_NONNUMERIC


class CSVDumper(abstract_dumper.AbstractDumper):
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
    def _(self, event: log_events.EpochMetrics) -> None:
        model_name = str(event.model_name)
        file_path = self.csv_path(model_name, event.source)
        with file_path.open('a') as log:
            writer = csv.writer(log, dialect=self._dialect)
            if not log.tell():
                writer.writerow(['Epoch', *event.metrics])
            writer.writerow([event.epoch, *event.metrics.values()])
        return

    def csv_path(self, model_name: str, source: str) -> pathlib.Path:
        """
        Return path to the csv file.

        Args:
            model_name: name of the model.
            source: source of the metrics.
        Returns:
            path to the csv file.
        """
        model_path = self.par_dir / str(model_name)
        model_path.mkdir(exist_ok=True)
        return (model_path / str(source)).with_suffix('.csv')

    def read_csv(self,
                 model_name: str,
                 source: str,
                 ) -> tuple[list[str], list[list[Any]]]:
        """
        Reads the CSV file associated with the given model and source.

        Args:
            model_name: name of the model.
            source: source of the metrics.

        Returns:
            column headers and the data as list of list (of float when using
                the default dialect).
        """
        file_path = self.csv_path(model_name, source)
        with file_path.open() as log:
            reader = csv.reader(log, dialect=self._dialect)
            columns = next(reader)
            return columns, list(reader)

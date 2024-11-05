import abc
import csv
import io
import pathlib
from typing import Optional

import pandas
import pandas as pd
from mypy.memprofile import defaultdict

from src.dry_torch import log_backend


class BaseCSVBackend(log_backend.LogBackend, abc.ABC):
    pass
    def plot(self,
             model_name: str,
             partition: str,
             metric: str) -> None:
        pass

class MemoryCSVBackend(BaseCSVBackend):
    """Logging backend implementation using SQLite with SQLAlchemy."""

    def __init__(self) -> None:
        self.dict_csv_str: dict[tuple[str, str], str] = defaultdict(str)

    def __call__(self,
                 model_name: str,
                 source: str,
                 partition: str,
                 epoch: int,
                 metrics: dict[str, float]) -> None:

        string_io = io.StringIO(self.dict_csv_str[(model_name, partition)])
        writer = csv.writer(string_io)
        if not string_io.tell():
            writer.writerow(['Source', 'Epoch', *metrics])
        writer.writerow([source, epoch, *metrics.values()])
        self.dict_csv_str[(model_name, partition)] = string_io.getvalue()
        string_io.close()
        return

    def read_log(self, model_name: str, partition: str) -> pd.DataFrame:
        csv_string = self.dict_csv_str[(model_name, partition)]
        if not csv_string:
            return pd.DataFrame()
        return pandas.read_csv(io.StringIO(csv_string))

class FileCSVBackend(BaseCSVBackend):
    """Logging backend implementation using SQLite with SQLAlchemy."""

    def __init__(self, exp_path: pathlib.Path) -> None:
        self.exp_path = exp_path

    def __call__(self,
                 model_name: str,
                 source: str,
                 partition: str,
                 epoch: int,
                 metrics: dict[str, float]) -> None:

        log_path = self._get_log_path(model_name, partition)
        with log_path.open('a') as log:
            writer = csv.writer(log)
            if not log.tell():
                writer.writerow(['Source', 'Epoch', *metrics])
            writer.writerow([source, epoch, *metrics.values()])
        return


    def _get_log_path(self, model_name: str, partition: str) -> pathlib.Path:
        model_dir = self.exp_path / model_name
        model_dir.mkdir(exist_ok=True, parents=True)
        return (model_dir / partition).with_suffix('.csv')

    def read_log(self, model_name: str, partition: str) -> pd.DataFrame:
        return pandas.read_csv(self._get_log_path(model_name, partition).open())


class CSVLog(log_backend.ExperimentLog):

    def create_log(self,
                   local_path: Optional[pathlib.Path],
                   exp_name: str) -> log_backend.LogBackend:
        if local_path is None:
            return MemoryCSVBackend()
        return FileCSVBackend(local_path)


def my_test():
    exp_log = CSVLog()
    backend1 = exp_log.create_log(None, exp_name='first')
    backend2 = exp_log.create_log(None, exp_name='second')
    backend1('model_name',
             'train',
             'split',
             1,
             {'loss': 0.1, 'accuracy': 0.9})
    backend2('another_model',
             'train',
             'split',
             1,
             {'loss': 0.1, 'accuracy': 0.9})
    print(backend1.read_log('model_name', 'split'))


my_test()

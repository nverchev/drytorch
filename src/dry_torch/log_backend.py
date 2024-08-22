import abc
import pathlib
from typing import Type

import sqlalchemy

class LogBackend(abc.ABC):
    """Abstract base class for logging backend implementations."""
    def __init__(self, exp_dir: pathlib.Path):
        self.exp_dir = exp_dir
        self.exp_name = exp_dir.name

    @abc.abstractmethod
    def log_metrics(self):
        ...


class SQLiteBackend(LogBackend):
    """Abstract base class for logging backend implementations."""

    def __init__(self, exp_name):
        super().__init__(exp_name)
        database_path = self.exp_dir / 'log.db'
        engine = f'sqlite://{database_path.as_posix()}'
        self.exp_name = exp_name
        self.engine = sqlalchemy.create_engine(engine)
        self.metadata = sqlalchemy.MetaData()

    def log_metrics(self):
        self.results = sqlalchemy.Table('results', self.metadata,
                                        sqlalchemy.Column('id',
                                                          sqlalchemy.Integer,
                                                          primary_key=True),
                                        sqlalchemy.Column('name',
                                                          sqlalchemy.String(
                                                              255)),
                                        sqlalchemy.Column('value',
                                                          sqlalchemy.Float),
                                        sqlalchemy.Column('timestamp',
                                                          sqlalchemy.DateTime,
                                                          default=sqlalchemy.func.now()))


log_backend: Type[LogBackend] = SQLiteBackend

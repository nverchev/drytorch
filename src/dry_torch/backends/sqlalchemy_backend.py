import datetime
import pathlib
from typing import Optional

import sqlalchemy
from sqlalchemy import orm

from dry_torch.log_backend import LogBackend, ExperimentLog


class Base(orm.DeclarativeBase):
    ...


class LoggedMetrics(orm.MappedAsDataclass, Base):
    """User class will be converted to a dataclass"""

    __tablename__ = 'logged_metrics'

    id: orm.Mapped[int] = orm.mapped_column(
        init=False, primary_key=True, autoincrement=True)
    experiment: orm.Mapped[str] = orm.mapped_column(index=True)
    model_name: orm.Mapped[str] = orm.mapped_column(index=True)
    source: orm.Mapped[str] = orm.mapped_column(index=True)
    partition: orm.Mapped[str] = orm.mapped_column(index=True)
    epoch: orm.Mapped[int] = orm.mapped_column()
    metric_name: orm.Mapped[str] = orm.mapped_column()
    metric_value: orm.Mapped[float] = orm.mapped_column()
    timestamp: orm.Mapped[datetime.datetime] = orm.mapped_column(
        init=False, insert_default=sqlalchemy.func.now())


class SQLiteBackend(LogBackend):
    """Logging backend implementation using SQLite with SQLAlchemy."""

    def __init__(self, engine: sqlalchemy.Engine, exp_name: str) -> None:
        self.engine = engine
        self.exp_name = exp_name
        self.Table = LoggedMetrics
        self.Session = orm.sessionmaker(bind=engine)
        Base.metadata.create_all(bind=engine)

    def __call__(self,
                 model_name: str,
                 source: str,
                 partition: str,
                 epoch: int,
                 metrics: dict[str, float]) -> None:
        session = self.Session()
        for i, (metric_name, value) in enumerate(metrics.items()):
            new_row = self.Table(experiment=self.exp_name,
                                 model_name=model_name,
                                 source=source,
                                 partition=partition,
                                 epoch=epoch,
                                 metric_name=metric_name,
                                 metric_value=value)
            session.add(new_row)

        session.commit()
        session.close()

    def print_views(self) -> None:
        distinct_combinations = self.Session().query(
            LoggedMetrics.model_name,
            LoggedMetrics.source,
            LoggedMetrics.partition
        ).where(LoggedMetrics.experiment == self.exp_name).distinct().all()
        print(distinct_combinations)
        return

    def plot(self,
             model_name: str,
             partition: str,
             metric: str) -> None:
        pass


class SQLiteLog(ExperimentLog):

    def __init__(self, storage='sqlite:///'):
        self.storage = storage

    def create_log(self,
                   local_path: Optional[pathlib.Path],
                   exp_name: str) -> LogBackend:
        if local_path is None:
            db_path = ':memory:'
        else:
            db_path = (local_path / 'log.db').absolute().as_posix()

        engine = sqlalchemy.create_engine(self.storage + db_path)
        return SQLiteBackend(engine, exp_name)


def my_test():
    exp_log = SQLiteLog()
    backend1 = exp_log.create_log(pathlib.Path(), exp_name='first')
    backend2 = exp_log.create_log(pathlib.Path().absolute().parent,
                                  exp_name='second')
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


my_test()

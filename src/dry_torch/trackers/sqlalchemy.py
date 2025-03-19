import datetime
import functools
from typing import Optional
from typing_extensions import override

import sqlalchemy
from sqlalchemy import orm

from dry_torch import log_events
from dry_torch import tracking


class Base(orm.DeclarativeBase):
    ...


class LoggedMetrics(orm.MappedAsDataclass, Base):
    """User class will be converted to a dataclass"""

    __tablename__ = 'logged_metrics'

    id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True
    )
    experiment: orm.Mapped[str] = orm.mapped_column(index=True)
    model_name: orm.Mapped[str] = orm.mapped_column(index=True)
    source: orm.Mapped[str] = orm.mapped_column(index=True)
    epoch: orm.Mapped[int] = orm.mapped_column()
    metric_name: orm.Mapped[str] = orm.mapped_column(index=True)
    metric_value: orm.Mapped[float] = orm.mapped_column()
    timestamp: orm.Mapped[datetime.datetime] = orm.mapped_column(
        init=False,
        insert_default=sqlalchemy.func.now()
    )


class SQLConnection(tracking.Tracker):
    """Tracker that dumps metrics into a SQL database."""

    def __init__(self,
                 drivername: str = 'sqlite',
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 host: Optional[str] = None,
                 database: str | None = 'logged_metrics.db') -> None:
        """
        Args:
            drivername: see sqlalchemy.engine.URL.create documentation
            username: see sqlalchemy.engine.URL.create documentation
            password: see sqlalchemy.engine.URL.create documentation
            host: see sqlalchemy.engine.URL.create documentation
            database: see sqlalchemy.engine.URL.create documentation.
        """
        super().__init__()
        self.url = sqlalchemy.URL.create(drivername,
                                         username=username,
                                         password=password,
                                         host=host,
                                         database=database)
        self.engine = sqlalchemy.create_engine(self.url)
        self.Session = orm.sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self._exp_name: str | None = None

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        self._exp_name = event.exp_name
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self._exp_name = None
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        if self._exp_name is None:
            raise RuntimeError('Access outside experiment scope.')
        with self.Session() as session:
            for i, (metric_name, value) in enumerate(event.metrics.items()):
                new_row = LoggedMetrics(experiment=self._exp_name,
                                        model_name=event.model_name,
                                        source=event.source,
                                        epoch=event.epoch,
                                        metric_name=metric_name,
                                        metric_value=value)
                session.add(new_row)

            session.commit()
        return

    def create_view(self):
        ...

    def get_metrics(
            self,
            model_name: str,
            source: str,
            metric_name: str,
            exp_name: Optional[str] = None,
    ) -> tuple[list[int], list[float]]:
        """
        Filter the dataset to obtain metric values.

        Args:
            exp_name: name of the experiment. Defaults to current one.
            model_name: name of the model
            metric_name: the name of the requested metric
            source: name of the source
        Returns:
            epochs and values of the requested metric
        """
        if exp_name is not None:
            if self._exp_name is None:
                raise RuntimeError('Access outside experiment scope.')
            exp_name = self._exp_name

        with self.Session() as session:
            # Get all distinct metric names for this experiment
            query = sqlalchemy.select(LoggedMetrics.epoch,
                                      LoggedMetrics.metric_value)
            query = query.where(LoggedMetrics.model_name == model_name,
                                LoggedMetrics.source == source,
                                LoggedMetrics.metric_name == metric_name,
                                LoggedMetrics.experiment == exp_name)

            epochs = list[int]()
            values = list[float]()
            for row in session.execute(query):
                epochs.append(row[0])
                values.append(row[1])
            return epochs, values

import datetime
import functools
from typing import Optional
from typing_extensions import override

import sqlalchemy
from sqlalchemy import orm, select, case, String, Float, Integer, func
from sqlalchemy import URL

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
                 database: Optional[str] = None) -> None:
        """
        Args:
            drivername: see sqlalchemy.engine.URL.create documentation
            username: see sqlalchemy.engine.URL.create documentation
            password: see sqlalchemy.engine.URL.create documentation
            host: see sqlalchemy.engine.URL.create documentation
            database: see sqlalchemy.engine.URL.create documentation
        """
        super().__init__()
        self.url = URL.create(drivername,
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
    def _(self, event: log_events.EpochMetrics) -> None:
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
            exp_name: Optional[str] = None,
    ) -> tuple[list[int], dict[str, list[float]]]:
        """
        Get a pivoted view of metrics.

        Args:
            exp_name: name of the experiment. Defaults to current one.
            model_name: name of the model
            source: name of the source

        Returns:
            epochs and metrics
        """
        if exp_name is not None:
            if self._exp_name is None:
                raise RuntimeError('Access outside experiment scope.')
            exp_name = self._exp_name

        with self.Session() as session:
            # Get all distinct metric names for this experiment
            query = select(LoggedMetrics.epoch,
                           LoggedMetrics.metric_name,
                           LoggedMetrics.metric_value)
            if exp_name is not None:
                query = query.where(
                    LoggedMetrics.model_name == model_name,
                )
            if model_name is not None:
                query = query.where(
                    LoggedMetrics.model_name == model_name,
                )
            if source is not None:
                query = query.where(
                    LoggedMetrics.source == source,
                )
            epochs = list[int]()
            metrics = dict[str, list[float]]()
            for row in session.execute(query):
                if not epochs or epochs[-1] != row[0]:
                    epochs.append(row[0])
                metrics.setdefault(row[1], []).append(row[2])
            assert {len(values) for values in metrics.values()} == {len(epochs)}
            return epochs, metrics

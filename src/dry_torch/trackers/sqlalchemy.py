"""Table classes and tracker that uses sqlalchemy to track metrics."""

from __future__ import annotations

import datetime
import functools
from typing import Optional, cast
from typing_extensions import override
import warnings

import sqlalchemy
from sqlalchemy import orm

from dry_torch import log_events
from dry_torch import tracking
from dry_torch import exceptions

class Base(orm.DeclarativeBase):
    ...


class Run(orm.MappedAsDataclass, Base):
    """
    Table for runs.

    A new run is created for each experiment scope, unless specified.

    Attributes:
        run_id: unique id for the table.
        experiments: list of experiments in the same run
        sources: list of sources from experiments
    """
    __tablename__ = 'runs'
    run_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    experiments: orm.Mapped[list[Experiment]] = orm.relationship(
        init=False,
        cascade='all, delete-orphan',
    )
    sources: orm.Mapped[list[Source]] = orm.relationship(
        init=False,
        cascade='all, delete-orphan',
    )


class Experiment(orm.MappedAsDataclass, Base):
    """
    Table for experiments.

    Attributes:
        experiment_id: unique id for the table.
        experiment_name_short: short format for the experiment's name.
        experiment_name_long: long format for the experiment's name.
        run_id: id of the run for the experiment.
        run: the entry for the run for the experiment.
    """
    __tablename__ = 'experiments'
    experiment_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    experiment_name_short: orm.Mapped[str] = orm.mapped_column(index=True)
    experiment_name_long: orm.Mapped[str] = orm.mapped_column()
    run_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Run.run_id),
        init=False,
    )
    run: orm.Mapped[Run] = orm.relationship(back_populates=Run.experiments.key)


class Source(orm.MappedAsDataclass, Base):
    """
    Table for sources.

    Attributes:
        source_id: unique id for the table.
        model_name_short: short format for the associated model's name.
        source_name_long: long format for the source's name.
        model_name_short: short format for the associated model's name.
        source_name_long: long format for the source's name.
        run_id: id of the run for the experiment.
        run: the entry for the run for the experiment.
        logs: list of logs originating from the source.
    """
    __tablename__ = 'sources'
    source_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    model_name_short: orm.Mapped[str] = orm.mapped_column(index=True)
    source_name_short: orm.Mapped[str] = orm.mapped_column(index=True)
    model_name_long: orm.Mapped[str] = orm.mapped_column()
    source_name_long: orm.Mapped[str] = orm.mapped_column()
    run_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Run.run_id),
        init=False,
    )
    run: orm.Mapped[Run] = orm.relationship(back_populates=Run.sources.key)
    logs: orm.Mapped[list[Log]] = orm.relationship(
        init=False,
        cascade='all, delete-orphan',
    )


class Log(orm.MappedAsDataclass, Base):
    """
    Table for the logs of the metrics.

    Attributes:
        log_id: unique id for the table.
        source_id: the id of the source creating the log.
        source: the entry for source creating the log.
        epoch: number of epochs the model has been trained.
        metric_name: the name of the metric.
        value: the value of the metric.
        created_at: the timestamp for the entry creation.
    """
    __tablename__ = 'logs'

    log_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    source_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Source.source_id),
        init=False,
    )
    source: orm.Mapped[Source] = orm.relationship(
        back_populates=Source.logs.key,
    )
    epoch: orm.Mapped[int] = orm.mapped_column(index=True)
    metric_name: orm.Mapped[str] = orm.mapped_column(index=True)
    value: orm.Mapped[float] = orm.mapped_column()
    created_at: orm.Mapped[datetime.datetime] = orm.mapped_column(
        init=False,
        insert_default=sqlalchemy.func.now()
    )


class SQLConnection(tracking.Tracker):
    """
    Tracker that creates a connection to a SQL databases using sqlalchemy.

    Class Attributes:
        default_url: by default, it creates a local sqlite database.

    Attributes:
        engine: the sqlalchemy Engine for the connection.
        Session: the Session class to initiate a sqlalchemy session.
        resume_run: resume the previous run instead of create a new one.

    """
    default_url = sqlalchemy.URL.create('sqlite', database='metrics.db')

    def __init__(self,
                 engine: Optional[sqlalchemy.Engine] = None,
                 resume_run: bool = False) -> None:

        """
        Args:
            engine: engine for the sqlalchemy session. Default uses default_url.
            resume_run: whether to resume the previous run.
        """
        super().__init__()
        self.engine = engine or sqlalchemy.create_engine(self.default_url)
        Base.metadata.create_all(bind=self.engine)
        self.Session = orm.sessionmaker(bind=self.engine)
        self.resume_run = resume_run
        self._run: Run | None = None
        self._sources = dict[str, Source]()

    @property
    def run(self) -> Run:
        """The current run."""
        if self._run is None:
            raise exceptions.AccessOutsideScopeError()
        return self._run

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        exp_name_short = format(event.exp_name, 's')
        with self.Session() as session:
            if self.resume_run:
                run_or_none = self._get_last_run(exp_name_short)
                if run_or_none is None:
                    msg = 'SQLConnection: No previous runs. Starting a new one.'
                    warnings.warn(msg)
                self._run = run_or_none
            if self._run is None:
                self._run = Run()

            experiment = Experiment(experiment_name_short=exp_name_short,
                                    experiment_name_long=event.exp_name,
                                    run=self.run)
            session.add(experiment)
            session.commit()

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self._run = None
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        with self.Session() as session:
            session.merge(self.run)
            if event.source in self._sources:
                source = self._sources[event.source]
                session.merge(source)
            else:
                source = self._sources.setdefault(event.source, Source(
                    model_name_long=event.model_name,
                    model_name_short=format(event.model_name, 's'),
                    source_name_long=event.source,
                    source_name_short=format(event.source, 's'),
                    run=self.run,
                ))
            for i, (metric_name, value) in enumerate(event.metrics.items()):
                new_row = Log(source=source,
                              epoch=event.epoch,
                              metric_name=metric_name,
                              value=value)
                session.add(new_row)
            session.commit()
        return

    def _get_run_metrics(
            self,
            sources: list[Source],
            max_epoch: Optional[int] = None,
    ) -> tuple[list[int], dict[str, list[float]]]:
        with self.Session() as session:
            query = session.query(Log).where(
                Log.source_id.in_((source.source_id for source in sources)),
            )
            if max_epoch is not None:
                query = query.where(Log.epoch <= max_epoch)
            epochs = list[int]()
            named_metric_values = dict[str, list[float]]()
            for log in query:
                log = cast(Log, log)  # fixing wrong annotation
                epoch = log.epoch
                if not epochs or epochs[-1] != epoch:
                    epochs.append(epoch)
                values = named_metric_values.setdefault(log.metric_name, [])
                values.append(log.value)
            return epochs, named_metric_values

    def _find_run_sources(self,
                          model_name: str,
                          run: Run) -> dict[str, list[Source]]:
        with self.Session() as session:
            query = session.query(Source).where(
                Source.run_id.is_(run.run_id),
                Source.model_name_short.is_(model_name),
            )
            named_sources = dict[str, list[Source]]()
            for source in query:
                source = cast(Source, source)  # fixing wrong annotation
                short_name = str(source.source_name_short)
                sources = named_sources.setdefault(short_name, [])
                sources.append(source)
            return named_sources

    def _get_last_run(self, exp_name: str) -> Run | None:
        with self.Session() as session:
            query = sqlalchemy.select(Run).join(Experiment).where(
                Experiment.experiment_name_short.is_(exp_name)
            ).order_by(Run.run_id.desc()).limit(1)
            return session.scalars(query).first()

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

        Returns:
            current epochs and named metric values by source.

        Raises:
            RuntimeError if called outside the experiment scope.
        """
        model_name_short = format(model_name, 's')
        last_sources = self._find_run_sources(model_name_short, self.run)
        out = dict[str, tuple[list[int], dict[str, list[float]]]]()
        for source_name, run_sources in last_sources.items():
            out[source_name] = self._get_run_metrics(run_sources,
                                                     max_epoch=max_epoch)
        return out

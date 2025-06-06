"""Module containing sqlalchemy Table classes and a tracker to track metrics."""

from __future__ import annotations

import datetime
import functools
from typing import Optional, cast
from typing_extensions import override
import warnings

import sqlalchemy
from sqlalchemy import orm

from dry_torch import exceptions
from dry_torch import log_events
from dry_torch.trackers import base_classes


class Base(orm.DeclarativeBase):
    ...


class Run(orm.MappedAsDataclass, Base):
    """
    Table for runs.

    A new run is created for each experiment scope, unless specified.

    Attributes:
        run_id: the unique id for the table.
        experiments: the list of experiments in the same run
        sources: the list of sources from experiments
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
        experiment_id: the unique id for the table.
        experiment_name: the experiment's name.
        experiment_version: the experiment's version.
        run_id: the id of the run for the experiment.
        run: the entry for the run for the experiment.
    """
    __tablename__ = 'experiments'
    experiment_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    experiment_name: orm.Mapped[str] = orm.mapped_column(index=True)
    experiment_version: orm.Mapped[str] = orm.mapped_column()
    run_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Run.run_id),
        init=False,
    )
    run: orm.Mapped[Run] = orm.relationship(back_populates=Run.experiments.key)


class Source(orm.MappedAsDataclass, Base):
    """
    Table for sources.

    Attributes:
        source_id: the unique id for the table.
        model_name: the model's name.
        model_version: the model's version.
        source_name: the source's name.
        source_version: the source's version.
        run_id: the id of the run for the experiment.
        run: the entry for the run for the experiment.
        logs: the list of logs originating from the source.
    """
    __tablename__ = 'sources'
    source_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    model_name: orm.Mapped[str] = orm.mapped_column(index=True)
    model_version: orm.Mapped[str] = orm.mapped_column()
    source_name: orm.Mapped[str] = orm.mapped_column(index=True)
    source_version: orm.Mapped[str] = orm.mapped_column()
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
        log_id: the unique id for the table.
        source_id: the id of the source creating the log.
        source: the entry for source creating the log.
        epoch: the number of epochs the model has been trained.
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


class SQLConnection(base_classes.MetricLoader):
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
            engine: the engine for the session. Default uses default_url.
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
    def clean_up(self) -> None:
        self._run = None
        self._sources = {}
        return

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        with self.Session() as session:
            if self.resume_run:
                run_or_none = self._get_last_run(event.exp_name)
                if run_or_none is None:
                    msg = 'SQLConnection: No previous runs. Starting a new one.'
                    warnings.warn(msg)

                self._run = run_or_none

            if self._run is None:
                self._run = Run()

            experiment = Experiment(experiment_name=event.exp_name,
                                    experiment_version=event.exp_version,
                                    run=self.run)
            session.add(experiment)
            session.commit()

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.CallModel) -> None:
        with self.Session() as session:
            run = session.merge(self.run)
            source = Source(
                model_name=event.model_name,
                model_version=event.model_version,
                source_name=event.source_name,
                source_version=event.source_version,
                run=run,
            )
            session.add(source)
            session.commit()
            self._sources[event.source_name] = source
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.Metrics) -> None:
        with self.Session() as session:
            source = session.merge(self._sources[event.source_name])
            for i, (metric_name, value) in enumerate(event.metrics.items()):
                new_row = Log(source=source,
                              epoch=event.epoch,
                              metric_name=metric_name,
                              value=value)
                session.add(new_row)

            session.commit()
        return super().notify(event)

    def _find_sources(self, model_name: str) -> dict[str, list[Source]]:
        with self.Session() as session:
            run = session.merge(self.run)
            query = session.query(Source).where(
                Source.run_id.is_(run.run_id),
                Source.model_name.is_(model_name),
            )
            named_sources = dict[str, list[Source]]()
            for source in query:
                source = cast(Source, source)  # fixing wrong annotation
                sources = named_sources.setdefault(source.source_name, [])
                sources.append(source)

            if not named_sources:
                msg = f'No sources for model {model_name}.'
                raise exceptions.TrackerException(self, msg)

            return named_sources

    def _get_last_run(self, exp_name: str) -> Run | None:
        with self.Session() as session:
            query = sqlalchemy.select(Run).join(Experiment).where(
                Experiment.experiment_name.is_(exp_name)
            ).order_by(Run.run_id.desc()).limit(1)
            return session.scalars(query).first()

    def _get_run_metrics(
            self,
            sources: list[Source],
            max_epoch: int,
    ) -> base_classes.HistoryMetrics:
        with self.Session() as session:
            sources = [session.merge(source) for source in sources]
            query = session.query(Log).where(
                Log.source_id.in_((source.source_id for source in sources)),
            )
            if max_epoch != -1:
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

        for values in named_metric_values.values():
            if len(values) != len(epochs):
                msg = 'Missing or multiple entries given metric name and epoch.'
                raise exceptions.TrackerException(self, msg)

        return epochs, named_metric_values

    def _load_metrics(self,
                      model_name: str,
                      max_epoch: int = -1) -> base_classes.SourcedMetrics:
        last_sources = self._find_sources(model_name)
        out: base_classes.SourcedMetrics = {}
        for source_name, run_sources in last_sources.items():
            out[source_name] = self._get_run_metrics(run_sources,
                                                     max_epoch=max_epoch)

        return out

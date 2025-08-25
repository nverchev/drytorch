"""Module containing sqlalchemy Table classes and a tracker to track metrics."""

from __future__ import annotations

import datetime
import functools

from typing import Final, cast

import sqlalchemy

from sqlalchemy import orm, sql
from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.trackers import base_classes


reg = orm.registry()


@reg.mapped_as_dataclass
class Experiment:
    """Table for experiments.

    Attributes:
        row_id: the unique id for the table.
        experiment_name: the experiment's name.
        runs: the entry for the run for the experiment.
        tags: the list of tags for the experiment.
    """

    __tablename__ = 'experiments'
    row_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    experiment_name: orm.Mapped[str] = orm.mapped_column(index=True)
    tags: orm.Mapped[list[Tags]] = orm.relationship(
        init=False,
        cascade='all, delete-orphan',
    )

    runs: orm.Mapped[list[Run]] = orm.relationship(
        init=False,
        cascade='all, delete-orphan',
    )


@reg.mapped_as_dataclass
class Tags:
    """Table for tags for experiments.

    Attributes:
        row_id: the unique id for the table.
        experiment_id: the id for the experiment.
        text: a tag for the experiment.
    """

    __tablename__ = 'tags'
    row_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    experiment_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Experiment.row_id),
        init=False,
    )
    experiment: orm.Mapped[Experiment] = orm.relationship(
        back_populates=Experiment.tags.key
    )
    text: orm.Mapped[str] = orm.mapped_column(index=True)


@reg.mapped_as_dataclass
class Run:
    """Table for runs.

    A new run is created for each experiment scope, unless specified.

    Attributes:
        row_id: the unique id for the table.
        run_id: global identifier for the run.
        run_ts: the run's timestamp.
        experiment_id: the id of the experiment for the run.
        experiment: the entry for the experiment for the run.
        sources: the list of sources from experiments
    """

    __tablename__ = 'runs'
    row_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    run_id: orm.Mapped[str] = orm.mapped_column(index=True)
    run_ts: orm.Mapped[datetime.datetime] = orm.mapped_column()
    experiment_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Experiment.row_id),
        init=False,
    )
    experiment: orm.Mapped[Experiment] = orm.relationship(
        back_populates=Experiment.runs.key
    )

    sources: orm.Mapped[list[Source]] = orm.relationship(
        init=False,
        cascade='all, delete-orphan',
    )


@reg.mapped_as_dataclass
class Source:
    """Table for sources.

    Attributes:
        row_id: the unique id for the table.
        model_name: the model's name.
        model_ts: the model's timestamp.
        source_name: the source's name.
        source_ts: the source's timestamp.
        run_table_id: the table id for the current experiment's run.
        run: the entry for the current experiment's run.
        logs: the list of logs originating from the source.
    """

    __tablename__ = 'sources'
    row_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    model_name: orm.Mapped[str] = orm.mapped_column(index=True)
    model_ts: orm.Mapped[datetime.datetime] = orm.mapped_column()
    source_name: orm.Mapped[str] = orm.mapped_column(index=True)
    source_ts: orm.Mapped[datetime.datetime] = orm.mapped_column()
    run_table_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Run.row_id),
        init=False,
    )
    run: orm.Mapped[Run] = orm.relationship(back_populates=Run.sources.key)
    logs: orm.Mapped[list[Log]] = orm.relationship(
        init=False,
        cascade='all, delete-orphan',
    )


@reg.mapped_as_dataclass
class Log:
    """Table for the logs of the metrics.

    Attributes:
        row_id: the unique id for the table.
        source_id: the id of the source creating the log.
        source: the entry for the source creating the log.
        epoch: the number of epochs the model has been trained.
        metric_name: the name of the metric.
        value: the value of the metric.
        created_at: the timestamp for the entry creation.
    """

    __tablename__ = 'logs'

    row_id: orm.Mapped[int] = orm.mapped_column(
        init=False,
        primary_key=True,
        autoincrement=True,
    )
    source_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey(Source.row_id),
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
        insert_default=sql.func.now(),
    )


class SQLConnection(base_classes.MetricLoader):
    """Tracker that creates a connection to a SQL database using sqlalchemy.

    Class Attributes:
        default_url: by default, it creates a local sqlite database.

    Attributes:
        engine: the sqlalchemy Engine for the connection.
        session_factory: the Session class to initiate a sqlalchemy session.
    """

    default_url = sqlalchemy.URL.create('sqlite', database='metrics.db')

    def __init__(
        self,
        engine: sqlalchemy.Engine | None = None,
    ) -> None:
        """Constructor.

        Args:
            engine: the engine for the session. Default uses default_url.
        """
        super().__init__()
        self.engine: Final = engine or sqlalchemy.create_engine(
            self.default_url
        )
        reg.metadata.create_all(bind=self.engine)
        self.session_factory: Final = orm.sessionmaker(bind=self.engine)
        self._run: Run | None = None
        self._sources: Final = dict[str, Source]()
        return

    @property
    def run(self) -> Run:
        """The current run."""
        if self._run is None:
            raise exceptions.AccessOutsideScopeError()

        return self._run

    @override
    def clean_up(self) -> None:
        self._run = None
        self._sources.clear()
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        with self.session_factory() as session:
            experiment = Experiment(
                experiment_name=event.exp_name,
            )

            for tag_str in event.tags:
                tag = Tags(text=tag_str, experiment=experiment)
                session.add(tag)
            self._run = Run(event.run_id, event.run_ts, experiment)
            session.add(experiment)
            session.commit()

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ActorRegistrationEvent) -> None:
        with self.session_factory() as session:
            run = session.merge(self.run)
            source = Source(
                model_name=event.model_name,
                model_ts=event.model_ts,
                source_name=event.actor_name,
                source_ts=event.actor_ts,
                run=run,
            )
            session.add(source)
            session.commit()
            self._sources[event.actor_name] = source
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.MetricEvent) -> None:
        with self.session_factory() as session:
            if event.source_name not in self._sources:
                msg = f'Source {event.source_name} has not been registered.'
                raise exceptions.TrackerError(self, msg)

            source = session.merge(self._sources[event.source_name])
            for metric_name, value in event.metrics.items():
                new_row = Log(
                    source=source,
                    epoch=event.epoch,
                    metric_name=metric_name,
                    value=value,
                )
                session.add(new_row)

            session.commit()
        return super().notify(event)

    def _find_sources(self, model_name: str) -> dict[str, list[Source]]:
        with self.session_factory() as session:
            run = session.merge(self.run)
            query = (
                session.query(Source)
                .join(Source.run)
                .where(
                    Run.run_id.is_(run.run_id),
                    Source.model_name.is_(model_name),
                )
            )
            named_sources = dict[str, list[Source]]()
            for source in query:
                source = cast(Source, source)  # fixing wrong annotation
                sources = named_sources.setdefault(source.source_name, [])
                sources.append(source)

            if not named_sources:
                msg = f'No sources for model {model_name}.'
                raise exceptions.TrackerError(self, msg)

            return named_sources

    def _get_run_metrics(
        self,
        sources: list[Source],
        max_epoch: int,
    ) -> base_classes.HistoryMetrics:
        with self.session_factory() as session:
            sources = [session.merge(source) for source in sources]
            query = session.query(Log).where(
                Log.source_id.in_(source.row_id for source in sources),
            )
            if max_epoch != -1:
                query = query.where(Log.epoch <= max_epoch)

            named_epochs = dict[str, list[int]]()
            named_metric_values = dict[str, list[float]]()

            for log in query:
                log = cast(Log, log)  # fixing wrong annotation
                epochs = named_epochs.setdefault(log.metric_name, [])
                epochs.append(log.epoch)
                values = named_metric_values.setdefault(log.metric_name, [])
                values.append(log.value)

        name = ''
        epochs = list[int]()
        for next_name, next_epochs in named_epochs.items():
            if epochs and epochs != next_epochs:
                msg = f'{name} and {next_name} logs refer to different epochs.'
                raise exceptions.TrackerError(self, msg)
            epochs = next_epochs
            name = next_name

        return epochs, named_metric_values

    def _load_metrics(
        self, model_name: str, max_epoch: int = -1
    ) -> base_classes.SourcedMetrics:
        last_sources = self._find_sources(model_name)
        out: base_classes.SourcedMetrics = {}
        for source_name, run_sources in last_sources.items():
            out[source_name] = self._get_run_metrics(
                run_sources, max_epoch=max_epoch
            )

        return out

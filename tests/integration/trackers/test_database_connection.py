"""Tests for SQLConnection focusing on error conditions and edge cases."""

import gc
import pathlib
import threading

from collections.abc import Generator

import pytest


try:
    import sqlalchemy

    from sqlalchemy import exc as sqlalchemy_exc
except ImportError:
    pytest.skip('sqlalchemy not available', allow_module_level=True)
    raise

from drytorch.core import log_events
from drytorch.trackers import sqlalchemy as sql_models
from drytorch.trackers.sqlalchemy import Experiment, Source, SQLConnection


@pytest.fixture(autouse=True, scope='module')
def start_experiment(run) -> Generator[None, None, None]:
    """Create an experimental scope for the tests."""
    yield
    return


@pytest.fixture
def event_workflow(
    start_experiment_event,
    model_registration_event,
    load_model_event,
    start_training_event,
    actor_registration_event,
    start_epoch_event,
    iterate_batch_event,
    metrics_event,
    end_epoch_event,
    update_learning_rate_event,
    pause_experiment_event,
    continue_experiment_event,
    terminated_training_event,
    end_training_event,
    start_test_event,
    end_test_event,
    save_model_event,
    stop_experiment_event,
) -> tuple[log_events.Event, ...]:
    """Yields events in typical order of execution."""
    event_tuple = (
        start_experiment_event,
        model_registration_event,
        load_model_event,
        start_training_event,
        actor_registration_event,
        start_epoch_event,
        iterate_batch_event,
        metrics_event,
        end_epoch_event,
        update_learning_rate_event,
        pause_experiment_event,
        continue_experiment_event,
        terminated_training_event,
        end_training_event,
        start_test_event,
        end_test_event,
        save_model_event,
        stop_experiment_event,
    )
    return event_tuple


class TestSQLConnection:
    """Tests for the SQLConnection class."""

    @pytest.fixture
    def tracker(self, tmp_path) -> Generator[SQLConnection, None, None]:
        """Set up the instance."""
        engine = sqlalchemy.create_engine('sqlite:///:memory:')
        tracker = SQLConnection(engine=engine)
        yield tracker

        tracker.clean_up()
        gc.collect()
        engine.dispose()
        return

    def test_database_connection_failure(self) -> None:
        """Test behavior when the database connection fails."""
        # example case of connection failure
        invalid_engine = sqlalchemy.create_engine(
            'sqlite:///invalid/path/database.db'
        )

        with pytest.raises(sqlalchemy_exc.OperationalError):
            _ = SQLConnection(engine=invalid_engine)

    def test_concurrent_database_access(self, tmp_path, event_workflow) -> None:
        """Test two tracker instances accessing a database concurrently."""
        database = tmp_path / 'test_db.db'
        database = tmp_path / 'test_db.db'
        engine = sqlalchemy.create_engine(
            f'sqlite:///{database}', poolclass=sqlalchemy.pool.NullPool
        )
        tracker1 = SQLConnection(engine=engine)
        tracker2 = SQLConnection(engine=engine)
        succeeded = list[SQLConnection]()
        errors = []

        def _run_experiment(tracker):
            try:
                for event in event_workflow:
                    tracker.notify(event)
                succeeded.append(tracker)
            except Exception as e:
                errors.append(e)

        thread1 = threading.Thread(target=_run_experiment, args=(tracker1,))
        thread2 = threading.Thread(target=_run_experiment, args=(tracker2,))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        tracker1.clean_up()
        tracker2.clean_up()
        gc.collect()
        engine.dispose()
        assert len(succeeded) == 2

    def test_session_rollback_on_error(
        self,
        tracker,
        mocker,
        start_experiment_event,
        actor_registration_event,
    ) -> None:
        """Test that sessions are properly rolled back on errors."""

        def _raise_integrity_error(_):
            raise sqlalchemy_exc.IntegrityError('', '', ValueError())

        tracker.notify(start_experiment_event)
        tracker.notify(actor_registration_event)
        mocker.patch.object(
            sqlalchemy.orm.Session,  # type: ignore
            'add',
            side_effect=_raise_integrity_error,
        )
        with pytest.raises(sqlalchemy_exc.IntegrityError):
            tracker.notify(actor_registration_event)

        # verify database state is consistent (no partial commits)
        with tracker.session_factory() as session:
            sources = session.query(Source).all()
            experiments = session.query(Experiment).all()
            # the second source should have been rolled back
            assert len(experiments) == 1
            assert len(sources) == 1

        gc.collect()


class TestSQLConnectionPauseContinue:
    """Tests SQLConnection run isolation across interleaved runs."""

    @pytest.fixture
    def tracker(
        self, tmp_path: pathlib.Path
    ) -> Generator[SQLConnection, None, None]:
        """Create a SQLConnection instance with SQLite file."""
        db_file = tmp_path / 'metrics.db'
        engine = sqlalchemy.create_engine(f'sqlite:///{db_file.as_posix()}')
        instance = SQLConnection(engine=engine)
        yield instance
        instance.close()
        engine.dispose()
        return

    def test_interleaved_pause_continue(
        self,
        tracker: SQLConnection,
        example_exp_name: str,
        start_experiment_event: log_events.StartExperimentEvent,
        pause_experiment_event: log_events.PauseExperimentEvent,
        continue_experiment_event: log_events.ContinueExperimentEvent,
        stop_experiment_event: log_events.StopExperimentEvent,
        start_experiment_event_b: log_events.StartExperimentEvent,
        stop_experiment_event_b: log_events.StopExperimentEvent,
        actor_registration_event: log_events.ActorRegistrationEvent,
        metrics_event_a1: log_events.MetricEvent,
        metrics_event_b: log_events.MetricEvent,
        metrics_event_a2: log_events.MetricEvent,
        run_b_id: str,
    ) -> None:
        """Verify SQLConnection keeps separate run rows and isolated logs."""
        run_a = start_experiment_event.run_id

        # Trigger
        tracker.notify(start_experiment_event)
        tracker.notify(actor_registration_event)
        tracker.notify(metrics_event_a1)
        tracker.notify(pause_experiment_event)

        tracker.notify(start_experiment_event_b)
        tracker.notify(actor_registration_event)
        tracker.notify(metrics_event_b)
        tracker.notify(stop_experiment_event_b)

        tracker.notify(continue_experiment_event)
        tracker.notify(metrics_event_a2)
        tracker.notify(stop_experiment_event)

        # Assert
        with tracker.session_factory() as session:
            exp_rows = session.scalars(
                sqlalchemy.select(sql_models.Experiment)
            ).all()
            run_rows = session.scalars(sqlalchemy.select(sql_models.Run)).all()

            logs_a = session.scalars(
                sqlalchemy.select(sql_models.Log)
                .join(sql_models.Source)
                .join(sql_models.Run)
                .where(sql_models.Run.run_id == run_a)
                .order_by(sql_models.Log.epoch)
            ).all()
            logs_b = session.scalars(
                sqlalchemy.select(sql_models.Log)
                .join(sql_models.Source)
                .join(sql_models.Run)
                .where(sql_models.Run.run_id == run_b_id)
                .order_by(sql_models.Log.epoch)
            ).all()

            run_ids = {r.run_id for r in run_rows}
            values_a = [round(log.value, 4) for log in logs_a]
            values_b = [round(log.value, 4) for log in logs_b]

        assert len(exp_rows) == 1
        assert exp_rows[0].experiment_name == example_exp_name
        assert run_ids == {run_a, run_b_id}
        assert values_a == [0.1, 0.05]
        assert values_b == [0.9]
        assert 0.9 not in values_a

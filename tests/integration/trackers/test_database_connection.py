"""Tests for SQLConnection focusing on error conditions and edge cases."""

import pytest


try:
    import sqlalchemy
except ImportError:
    pytest.skip('sqlalchemy not available', allow_module_level=True)
    raise

import threading

from sqlalchemy import exc as sqlalchemy_exc

from drytorch.core import log_events
from drytorch.trackers.sqlalchemy import Experiment, Source, SQLConnection


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
    def memory_engine(self) -> sqlalchemy.Engine:
        """Return memory engine."""
        return sqlalchemy.create_engine('sqlite:///:memory:')

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
        engine = sqlalchemy.create_engine(f'sqlite:///{database}')
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
        assert len(succeeded) == 2

    def test_session_rollback_on_error(
        self,
        memory_engine,
        mocker,
        start_experiment_event,
        actor_registration_event,
    ) -> None:
        """Test that sessions are properly rolled back on errors."""
        tracker = SQLConnection(engine=memory_engine)

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

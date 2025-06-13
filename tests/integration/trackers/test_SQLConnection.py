"""Tests for SQLConnection focusing on error conditions and edge cases."""

import pytest

import threading

import sqlalchemy
from sqlalchemy import exc as sqlalchemy_exc

from drytorch import log_events
from drytorch.trackers.sqlalchemy import Experiment
from drytorch.trackers.sqlalchemy import Source
from drytorch.trackers.sqlalchemy import SQLConnection


@pytest.fixture
def event_workflow(
        start_experiment_event,
        model_creation_event,
        load_model_event,
        start_training_event,
        call_model_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
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
        model_creation_event,
        load_model_event,
        start_training_event,
        call_model_event,
        start_epoch_event,
        iterate_batch_event,
        epoch_metrics_event,
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
    """"""

    @pytest.fixture
    def memory_engine(self) -> sqlalchemy.Engine:
        """Return memory engine."""
        return sqlalchemy.create_engine('sqlite:///:memory:')

    def test_database_connection_failure(self) -> None:
        """Test behavior when database connection fails."""
        # example case of connection failure
        invalid_engine = sqlalchemy.create_engine(
            'sqlite:///invalid/path/database.db'
        )

        with pytest.raises(sqlalchemy_exc.OperationalError):
            _ = SQLConnection(engine=invalid_engine)

    def test_concurrent_database_access(self, tmp_path, event_workflow) -> None:
        """Test two tracker instances accessing same database concurrently."""
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

        thread1 = threading.Thread(target=_run_experiment,
                                   args=(tracker1,))
        thread2 = threading.Thread(target=_run_experiment,
                                   args=(tracker2,))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        assert len(succeeded) == 2

    def test_session_rollback_on_error(self,
                                       memory_engine,
                                       mocker,
                                       start_experiment_event,
                                       call_model_event) -> None:
        """Test that sessions are properly rolled back on errors."""

        tracker = SQLConnection(engine=memory_engine)

        def _raise_integrity_error(_):
            raise sqlalchemy_exc.IntegrityError("", "", ValueError())

        tracker.notify(start_experiment_event)
        tracker.notify(call_model_event)
        mocker.patch.object(sqlalchemy.orm.Session,
                            'add',
                            side_effect=_raise_integrity_error)
        with pytest.raises(sqlalchemy_exc.IntegrityError):
            tracker.notify(call_model_event)

        # verify database state is consistent (no partial commits)
        with tracker.Session() as session:
            sources = session.query(Source).all()
            experiments = session.query(Experiment).all()
            # second source should have been rolled back
            assert len(experiments) == 1
            assert len(sources) == 1

    def test_resume_nonexistent_experiment(self,
                                           memory_engine,
                                           start_experiment_event) -> None:
        """Test resume_run behavior when no previous runs exist."""
        tracker = SQLConnection(engine=memory_engine, resume_run=True)
        start_experiment_event.exp_name = 'nonexistent'

        with pytest.warns(UserWarning, match='No previous runs'):
            tracker.notify(start_experiment_event)

        # should create new run despite resume_run=True
        assert tracker._run is not None

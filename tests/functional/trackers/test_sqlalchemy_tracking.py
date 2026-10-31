"""Functional tests for connecting to a SQL database."""

import dataclasses

from collections.abc import Generator

import pytest


try:
    import sqlalchemy
except ImportError:
    pytest.skip('sqlalchemy not available', allow_module_level=True)
    raise

from drytorch.trackers.sqlalchemy import (
    Experiment,
    Log,
    Run,
    Source,
    SQLConnection,
    Tags,
)
from drytorch.trackers.sqlalchemy import (
    TestSource as _TestSource,
)


class TestSQLConnectionFullCycle:
    """Complete ``SQLConnection`` session and tests it afterward."""

    @pytest.fixture
    def tracker(self, tmp_path) -> Generator[SQLConnection, None, None]:
        """Set up the instance."""
        db_file = tmp_path / 'metrics.db'
        url = f'sqlite:///{db_file.as_posix()}'
        engine = sqlalchemy.create_engine(url)
        tracker = SQLConnection(engine=engine)
        yield tracker

        engine.dispose()
        return

    @pytest.fixture(autouse=True)
    def full_cycle(self, tracker, event_workflow) -> None:
        """Run an example session."""
        for event in event_workflow:
            tracker.notify(event)

        return

    @pytest.fixture
    def resumed_tracker(
        self,
        tracker,
        start_experiment_event,
        stop_experiment_event,
    ) -> Generator[SQLConnection, None, None]:
        """Resume previous run."""
        resumed_event = dataclasses.replace(
            start_experiment_event, resumed=True
        )
        resumed_tracker = SQLConnection(engine=tracker.engine)
        resumed_tracker.notify(resumed_event)
        yield resumed_tracker

        resumed_tracker.notify(stop_experiment_event)
        return

    def test_database_schema_created_correctly(self, tracker) -> None:
        """Test all required database tables are created."""
        inspector = sqlalchemy.inspect(tracker.engine)
        table_names = inspector.get_table_names()
        experiments_columns = {
            col['name'] for col in inspector.get_columns('experiments')
        }
        tags_columns = {col['name'] for col in inspector.get_columns('tags')}
        runs_columns = {col['name'] for col in inspector.get_columns('runs')}
        sources_columns = {
            col['name'] for col in inspector.get_columns('sources')
        }
        logs_columns = {col['name'] for col in inspector.get_columns('logs')}
        test_sources_columns = {
            col['name'] for col in inspector.get_columns('test_sources')
        }

        expected_tables = {
            'runs',
            'experiments',
            'sources',
            'logs',
            'tags',
            'test_sources',
        }
        expected_experiment_columns = {
            'row_id',
            'experiment_name',
        }
        expected_tag_columns = {
            'row_id',
            'experiment_id',
            'text',
        }
        expected_run_columns = {
            'row_id',
            'run_id',
            'run_ts',
            'experiment_id',
        }
        expected_source_columns = {
            'row_id',
            'model_name',
            'model_ts',
            'source_name',
            'source_ts',
            'run_table_id',
        }
        expected_test_source_columns = {
            'row_id',
            'source_id',
        }
        expected_log_columns = {
            'row_id',
            'source_id',
            'epoch',
            'metric_name',
            'value',
            'created_at',
        }

        assert expected_tables.issubset(set(table_names))
        assert experiments_columns == expected_experiment_columns
        assert tags_columns == expected_tag_columns
        assert runs_columns == expected_run_columns
        assert sources_columns == expected_source_columns
        assert test_sources_columns == expected_test_source_columns
        assert logs_columns == expected_log_columns

    def test_experiment_workflow_stored_correctly(self, tracker) -> None:
        """Test the experiment workflow is correctly stored in the database."""
        with tracker.session_factory() as session:
            runs = session.query(Run).all()
            first_run = runs[-1]
            experiments = session.query(Experiment).all()
            first_experiment = experiments[0]
            tags = session.query(Tags).all()
            sources = session.query(Source).all()
            logs = session.query(Log).all()
            source_ids = {source.row_id for source in sources}

            # verify run was created and linked to experiment
            assert len(runs) == 1
            assert len(experiments) == 1
            assert first_experiment.row_id == first_run.experiment_id
            assert len(tags) == 1

            # verify sources were created and linked to run
            assert len(sources) == 1
            for source in sources:
                assert source.run_table_id == first_run.row_id

            # verify logs were created and linked to sources
            assert len(logs) > 0
            for log in logs:
                assert log.source_id in source_ids

    def test_epoch_metrics_stored_with_correct_values(
        self, tracker, example_epoch, example_named_metrics
    ) -> None:
        """Test metrics are correctly stored in the database."""
        with tracker.session_factory() as session:
            logs = session.query(Log).filter(Log.epoch == example_epoch).all()
            stored_metrics = {log.metric_name: log.value for log in logs}
            expected_metrics = example_named_metrics

            assert len(logs) == len(expected_metrics)
            for metric_name, expected_value in expected_metrics.items():
                assert metric_name in stored_metrics
                assert stored_metrics[metric_name] == expected_value

    def test_multiple_epochs_stored_separately(self, tracker) -> None:
        """Test that metrics from different epochs are stored separately."""
        with tracker.session_factory() as session:
            epochs = session.query(Log.epoch).all()
            epoch_values = [epoch[0] for epoch in epochs]

            # expected 3 training epochs based on event_workflow
            assert len(set(epoch_values)) == 3

    def test_multiple_sources_in_same_run(
        self,
        resumed_tracker,
        runner_registration_event,
    ) -> None:
        """Test multiple log sources are correctly tracked in the same run."""
        second_call_model = dataclasses.replace(
            runner_registration_event, runner_name='second_source'
        )
        resumed_tracker.notify(second_call_model)

        with resumed_tracker.session_factory() as session:
            sources = session.query(Source).all()
            run_ids = {source.run.run_id for source in sources}
            source_names = {source.source_name for source in sources}

            assert len(sources) == 2
            assert len(run_ids) == 1
            assert len(source_names) == 2

    def test_load_metrics_returns_correct_data(
        self, resumed_tracker, start_training_event
    ) -> None:
        """Test _load_metrics returns the stored data correctly."""
        model_name = start_training_event.model_name

        loaded_metrics = resumed_tracker._load_metrics(model_name)

        assert len(loaded_metrics) == 1  # only one source in the workflow
        for _, (epochs, metrics_dict) in loaded_metrics.items():
            assert len(epochs) > 0
            for _, values in metrics_dict.items():
                assert len(values) == len(epochs)

    def test_load_metrics_with_max_epoch_filter(
        self, resumed_tracker, example_epoch, start_training_event
    ) -> None:
        """Test _load_metrics respects max_epoch parameter."""
        model_name = start_training_event.model_name
        max_epoch = example_epoch

        loaded_metrics = resumed_tracker._load_metrics(
            model_name, max_epoch=max_epoch
        )

        for _, (epochs, _) in loaded_metrics.items():
            assert epochs == [max_epoch]

    def test_load_metrics_with_resumed_sources(
        self,
        resumed_tracker,
        runner_registration_event,
        example_epoch,
        metrics_event,
        start_training_event,
    ) -> None:
        """Test _load_metrics get data from two sources with the same name."""
        model_name = start_training_event.model_name
        resumed_tracker.notify(runner_registration_event)
        new_epoch_metrics = dataclasses.replace(metrics_event, epoch=8)
        resumed_tracker.notify(new_epoch_metrics)

        loaded_metrics = resumed_tracker._load_metrics(model_name)

        assert len(loaded_metrics) == 1  # source is resumed
        for _, (epochs, _) in loaded_metrics.items():
            assert example_epoch in epochs
            assert 8 in epochs

    def test_test_source_creation(
        self,
        resumed_tracker,
        runner_registration_event,
        test_registration_event,
    ) -> None:
        """Test that TestSource is created only for a test."""
        test_event = dataclasses.replace(
            test_registration_event, runner_name='my_test_runner'
        )
        resumed_tracker.notify(test_event)

        with resumed_tracker.session_factory() as session:
            runner_source = (
                session.query(Source)
                .where(
                    Source.source_name.is_(
                        runner_registration_event.runner_name
                    )
                )
                .first()
            )
            test_source = (
                session.query(Source)
                .where(Source.source_name.is_('my_test_runner'))
                .first()
            )
            assert runner_source is not None
            assert test_source is not None

            runner_ts = (
                session.query(_TestSource)
                .where(_TestSource.source_id.is_(runner_source.row_id))
                .first()
            )
            created_ts = (
                session.query(_TestSource)
                .where(_TestSource.source_id.is_(test_source.row_id))
                .first()
            )
            assert runner_ts is None
            assert created_ts is not None

    def test_load_test_metrics_returns_only_tests_of_current_run(
        self,
        resumed_tracker,
        test_registration_event,
        start_experiment_event,
        stop_experiment_event,
    ) -> None:
        """Test load_test_metrics returns only the tests of the current run."""
        model_name = test_registration_event.model_name
        test_event_run1 = dataclasses.replace(
            test_registration_event, runner_name='test_run1'
        )
        resumed_tracker.notify(test_event_run1)

        exp_name = start_experiment_event.exp_name
        test_metrics_1 = resumed_tracker.load_test_metrics(model_name)
        assert 'test_run1' in test_metrics_1[exp_name]

        run2_event = dataclasses.replace(start_experiment_event, run_id='run_2')
        run2_tracker = SQLConnection(engine=resumed_tracker.engine)
        run2_tracker.notify(run2_event)
        test_event_run2 = dataclasses.replace(
            test_registration_event, runner_name='test_run2'
        )
        run2_tracker.notify(test_event_run2)

        test_metrics_2 = run2_tracker.load_test_metrics(model_name)
        assert 'test_run2' in test_metrics_2[exp_name]
        assert 'test_run1' not in test_metrics_2[exp_name]
        run2_tracker.notify(stop_experiment_event)

    def test_load_test_metrics_loads_all_epochs(
        self,
        resumed_tracker,
        test_registration_event,
        metrics_event,
    ) -> None:
        """Test load_test_metrics loads all recorded test epochs."""
        model_name = test_registration_event.model_name
        source = test_registration_event.runner_name
        resumed_tracker.notify(test_registration_event)
        for ep in (1, 2, 3):
            event = dataclasses.replace(
                metrics_event,
                epoch=ep,
                model_name=model_name,
                source_name=source,
            )
            resumed_tracker.notify(event)

        test_metrics = resumed_tracker.load_test_metrics(model_name)
        exp_name = next(iter(test_metrics))
        assert test_metrics[exp_name][source][0] == [1, 2, 3]

    def test_delete_experiment_removes_from_load_test_metrics(
        self,
        tracker,
        start_experiment_event,
        test_registration_event,
        metrics_event,
        stop_experiment_event,
    ) -> None:
        """Test delete_experiment removes experiment from load_test_metrics."""
        model_name = test_registration_event.model_name
        exp1_name = 'exp_1'
        exp2_name = 'exp_2'
        for exp_name in (exp1_name, exp2_name):
            start_event = dataclasses.replace(
                start_experiment_event,
                exp_name=exp_name,
                run_id=f'{exp_name}_run',
            )
            tracker.notify(start_event)
            tracker.notify(test_registration_event)
            metric_evt = dataclasses.replace(
                metrics_event,
                model_name=model_name,
                source_name=test_registration_event.runner_name,
            )
            tracker.notify(metric_evt)
            tracker.notify(stop_experiment_event)

        test_metrics = tracker.load_test_metrics(model_name)
        assert exp1_name in test_metrics
        assert exp2_name in test_metrics

        tracker.delete_experiment(exp1_name)

        updated_metrics = tracker.load_test_metrics(model_name)
        assert exp1_name not in updated_metrics
        assert exp2_name in updated_metrics

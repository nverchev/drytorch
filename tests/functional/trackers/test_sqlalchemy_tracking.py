"""Functional tests for connecting to a SQL database."""

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
)


class TestSQLConnectionFullCycle:
    """Complete SQLConnection session and tests it afterward."""

    @pytest.fixture
    def tracker(self) -> SQLConnection:
        """Set up the instance."""
        engine = sqlalchemy.create_engine('sqlite:///:memory:')
        return SQLConnection(engine=engine)

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
        """Set up the instance."""
        resumed_tracker = SQLConnection(engine=tracker.engine, resume_run=True)
        resumed_tracker.notify(start_experiment_event)
        yield resumed_tracker

        resumed_tracker.notify(stop_experiment_event)
        return

    def test_database_schema_created_correctly(self, tracker):
        """Test all required database tables are created."""
        inspector = sqlalchemy.inspect(tracker.engine)
        table_names = inspector.get_table_names()
        # verify all expected tables exist
        expected_tables = {'runs', 'experiments', 'sources', 'logs'}
        assert expected_tables.issubset(set(table_names))

        # verify key columns exist
        runs_columns = {col['name'] for col in inspector.get_columns('runs')}
        assert 'run_id' in runs_columns

        experiments_columns = {
            col['name'] for col in inspector.get_columns('experiments')
        }
        expected_columns = {
            'experiment_id',
            'experiment_name',
            'experiment_ts',
            'run_id',
        }
        assert experiments_columns == expected_columns

        sources_columns = {
            col['name'] for col in inspector.get_columns('sources')
        }
        expected_columns = {
            'model_name',
            'model_ts',
            'run_id',
            'source_id',
            'source_name',
            'source_ts',
        }
        assert sources_columns == expected_columns

        logs_columns = {col['name'] for col in inspector.get_columns('logs')}
        expected_columns = {
            'log_id',
            'source_id',
            'epoch',
            'metric_name',
            'value',
            'created_at',
        }
        assert logs_columns == expected_columns

    def test_experiment_workflow_stored_correctly(self, tracker):
        """Test the experiment workflow is correctly stored in database."""
        with tracker.session_factory() as session:
            # verify run was created
            runs = session.query(Run).all()
            assert len(runs) == 1

            run = runs[-1]
            # verify experiment was created and linked to run
            experiments = session.query(Experiment).all()
            assert len(experiments) == 1

            first_experiment = experiments[0]
            assert first_experiment.run_id == run.run_id

            # verify sources were created and linked to run
            sources = session.query(Source).all()
            assert len(sources) == 1

            for source in sources:
                assert source.run_id == run.run_id

            # verify logs were created and linked to sources
            logs = session.query(Log).all()
            assert len(logs) > 0

            source_ids = {source.source_id for source in sources}
            for log in logs:
                assert log.source_id in source_ids

    def test_epoch_metrics_stored_with_correct_values(
        self, tracker, example_epoch, example_named_metrics
    ):
        """Test metrics are correctly stored in the database."""
        with tracker.session_factory() as session:
            logs = session.query(Log).filter(Log.epoch == example_epoch).all()
            # number of metric should match the number of entries.
            expected_metrics = example_named_metrics
            assert len(logs) == len(expected_metrics)

            # verify name and value
            stored_metrics = {log.metric_name: log.value for log in logs}
            for metric_name, expected_value in expected_metrics.items():
                assert metric_name in stored_metrics
                assert stored_metrics[metric_name] == expected_value

    def test_multiple_epochs_stored_separately(self, tracker):
        """Test that metrics from different epochs are stored separately."""
        with tracker.session_factory() as session:
            epochs = session.query(Log.epoch).all()
            epoch_values = [epoch[0] for epoch in epochs]

            # expected 3 training epochs based on event_workflow
            assert len(set(epoch_values)) == 3

    def test_resume_run_functionality(self, resumed_tracker,):
        """Test resume_run=True continues previous run."""
        with resumed_tracker.session_factory() as session:
            runs = session.query(Run).all()
            assert len(runs) == 1

    def test_multiple_sources_in_same_run(
        self,
        resumed_tracker,
        call_model_event,
    ):
        """Test multiple log sources are correctly tracked in same run."""
        second_call_model = call_model_event
        second_call_model.source_name = 'second_source'
        resumed_tracker.notify(second_call_model)
        with resumed_tracker.session_factory() as session:
            sources = session.query(Source).all()
            assert len(sources) == 2

            # all sources should belong to same run
            run_ids = {source.run_id for source in sources}
            assert len(run_ids) == 1

            # have different source names
            source_names = {source.source_name for source in sources}
            assert len(source_names) == 2

    def test_load_metrics_returns_correct_data(
        self, resumed_tracker, start_training_event
    ):
        """Test _load_metrics returns the stored data correctly."""
        model_name = start_training_event.model_name
        loaded_metrics = resumed_tracker._load_metrics(model_name)
        assert len(loaded_metrics) == 1  # only one source in workflow

        for _, (epochs, metrics_dict) in loaded_metrics.items():
            # epochs should be not empty
            assert len(epochs) > 0

            # each metric should have same number of values as epochs
            for _, values in metrics_dict.items():
                assert len(values) == len(epochs)

    def test_load_metrics_with_max_epoch_filter(
        self, resumed_tracker, example_epoch, start_training_event
    ):
        """Test _load_metrics respects max_epoch parameter."""
        model_name = start_training_event.model_name
        max_epoch = example_epoch  # only get the first epoch
        loaded_metrics = resumed_tracker._load_metrics(
            model_name, max_epoch=max_epoch
        )
        for _, (epochs, _) in loaded_metrics.items():
            assert epochs == [max_epoch]

    def test_load_metrics_with_resumed_sources(
        self,
        resumed_tracker,
        call_model_event,
        example_epoch,
        metrics_event,
        start_training_event,
    ):
        """Test _load_metrics get data from two sources with same name."""
        model_name = start_training_event.model_name
        resumed_tracker.notify(call_model_event)
        metrics_event.epoch = 8
        resumed_tracker.notify(metrics_event)
        loaded_metrics = resumed_tracker._load_metrics(model_name)
        assert len(loaded_metrics) == 1  # source is resumed

        for _, (epochs, _) in loaded_metrics.items():
            # epochs contain logs from current and previous run
            assert example_epoch in epochs
            assert 8 in epochs

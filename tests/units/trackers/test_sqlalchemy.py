"""Tests for the "sqlalchemy" module."""

import copy
import dataclasses
import importlib.util

import pytest

from drytorch.core import exceptions, log_events


if not importlib.util.find_spec('sqlalchemy'):
    pytest.skip('sqlalchemy not available', allow_module_level=True)

from collections.abc import Generator

import sqlalchemy

from drytorch.trackers.sqlalchemy import (
    Experiment,
    Log,
    Run,
    Source,
    SQLConnection,
    Tags,
    TestSource,
)


class TestSQLConnection:
    """Tests for the ``SQLConnection`` tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
        self.mock_engine = mocker.Mock()
        self.mock_context = mocker.Mock()
        self.mock_session = mocker.MagicMock()
        self.mock_session.__enter__.return_value = self.mock_context
        self.MockSession = mocker.Mock(return_value=self.mock_session)
        self.create_engine_mock = mocker.Mock(return_value=self.mock_engine)
        self.make_mock_session = mocker.Mock(return_value=self.MockSession)
        self.exp = mocker.create_autospec(Experiment, instance=True)
        self.exp.tags = []
        self.exp.runs = []
        self.log = mocker.create_autospec(Log, instance=True)
        self.run = mocker.create_autospec(Run, instance=True)
        self.source = mocker.create_autospec(Source, instance=True)
        self.tags = mocker.create_autospec(Tags, instance=True)
        self.mock_last_run = mocker.Mock()
        mock_where = self.mock_context.query.return_value.where.return_value
        mock_where.first.return_value = None
        mocker.patch('sqlalchemy.create_engine', self.create_engine_mock)
        mocker.patch('sqlalchemy.orm.sessionmaker', self.make_mock_session)
        mocker.patch('sqlalchemy.schema.MetaData.create_all')
        self.Experiment = mocker.patch(
            'drytorch.trackers.sqlalchemy.Experiment', return_value=self.exp
        )
        self.Log = mocker.patch(
            'drytorch.trackers.sqlalchemy.Log', return_value=self.log
        )
        self.Run = mocker.patch(
            'drytorch.trackers.sqlalchemy.Run', return_value=self.run
        )
        self.Source = mocker.patch(
            'drytorch.trackers.sqlalchemy.Source', return_value=self.source
        )
        self.Tags = mocker.patch(
            'drytorch.trackers.sqlalchemy.Tags', return_value=self.tags
        )
        self.TestSource = mocker.patch(
            'drytorch.trackers.sqlalchemy.TestSource',
            return_value=mocker.Mock(),
        )
        return

    @pytest.fixture
    def tracker(self) -> Generator[SQLConnection, None, None]:
        """Set up the instance."""
        tracker = SQLConnection()
        yield tracker

        tracker.clean_up()
        return

    @pytest.fixture
    def tracker_started(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[SQLConnection, None, None]:
        """Start the instance."""
        tracker.notify(start_experiment_mock_event)
        yield tracker

        tracker.notify(stop_experiment_mock_event)
        return

    def test_unrecorded_run_stores_nothing(
        self, tracker, start_experiment_mock_event, epoch_metrics_mock_event
    ) -> None:
        """Test an unrecorded run stores no run and no metric."""
        start_experiment_mock_event.record = False
        tracker.notify(start_experiment_mock_event)

        tracker.notify(epoch_metrics_mock_event)

        self.Run.assert_not_called()
        self.Log.assert_not_called()

    def test_cleanup(self, tracker_started) -> None:
        """Test correct cleaning up."""
        tracker_started.clean_up()

        assert self.mock_engine.dispose.call_count == 0
        assert tracker_started._run is None
        assert tracker_started._sources == {}

    def test_init_default(self, tracker) -> None:
        """Test initialization with default parameters."""
        self.create_engine_mock.assert_called_once_with(tracker.default_url)
        self.make_mock_session.assert_called_once_with(bind=self.mock_engine)

    def test_run_property_before_start_raises_exception(self, tracker) -> None:
        """Test run property raises exception before experiment start."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _ = tracker.run

    def test_notify_start_experiment_creates_new_run(
        self,
        tracker_started,
        start_experiment_mock_event,
    ) -> None:
        """Test start experiment notification creates new tables."""
        added = [call.args[0] for call in self.mock_context.add.call_args_list]
        assert tracker_started.run == self.run
        assert self.exp in added
        assert self.run in added

    def test_notify_start_stop_experiment(
        self,
        tracker_started,
        stop_experiment_mock_event,
    ) -> None:
        """Test stop experiment notification cleans up the state."""
        tracker_started.notify(stop_experiment_mock_event)
        assert tracker_started._run is None

    def test_notify_call_model(
        self,
        tracker_started,
        runner_registration_mock_event,
    ) -> None:
        """Test call model notification creates the source."""
        tracker_started.notify(runner_registration_mock_event)
        self.mock_context.add.assert_called_with(self.source)
        sources = tracker_started._sources
        assert runner_registration_mock_event.runner_name in sources

    def test_notify_metrics(
        self,
        tracker_started,
        runner_registration_mock_event,
        epoch_metrics_mock_event,
    ) -> None:
        """Test metrics notification creates log entries."""
        tracker_started.notify(runner_registration_mock_event)
        tracker_started.notify(epoch_metrics_mock_event)
        self.mock_context.merge.assert_called_with(self.source)
        self.mock_context.add.assert_called_with(self.log)

    def test_unknown_source(
        self,
        tracker_started,
        runner_registration_mock_event,
        epoch_metrics_mock_event,
    ) -> None:
        """Test metrics notification from an unknown source raises an error."""
        tracker_started.notify(runner_registration_mock_event)
        epoch_metrics_mock_event.source_name = 'unknown_source'
        with pytest.raises(exceptions.TrackerError):
            tracker_started.notify(epoch_metrics_mock_event)

    def test_find_sources_nonexistent_model(
        self, mocker, tracker_started
    ) -> None:
        """Test _find_sources for nonexistent model returns empty mapping."""
        mock_query = mocker.MagicMock()
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = [].__iter__()
        self.mock_context.query.return_value = mock_query

        assert tracker_started._find_sources('nonexistent_model') == {}

    def test_get_run_metrics(
        self,
        mocker,
        tracker_started,
    ) -> None:
        """Test getting multiple metrics from the same epoch."""
        mock_log = mocker.Mock()
        mock_log.epoch = 1
        mock_log.metric_name = 'test_model'
        mock_log.value = 2.0
        mock_log2 = mocker.Mock()
        mock_log2.epoch = 1
        mock_log2.metric_name = 'test_model_2'
        mock_log2.value = 4.0
        mock_log3 = mocker.Mock()
        mock_log3.epoch = 2
        mock_log3.metric_name = 'test_model'
        mock_log3.value = 3.0
        mock_log4 = mocker.Mock()
        mock_log4.epoch = 2
        mock_log4.metric_name = 'test_model_2'
        mock_log4.value = 5.0

        mock_list = [mock_log, mock_log2, mock_log3, mock_log4]
        mock_query = mocker.MagicMock()
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = mock_list.__iter__()
        self.mock_context.query.return_value = mock_query
        epochs, metrics = tracker_started._get_run_metrics([], -1)

        assert epochs == [1, 2]
        assert metrics['test_model'] == [2, 3]
        assert metrics['test_model_2'] == [4, 5]

    def test_get_run_wrong_metrics(
        self,
        mocker,
        tracker_started,
    ) -> None:
        """Test missing metric."""
        mock_log3 = mocker.Mock()
        mock_log3.epoch = 1
        mock_log3.metric_name = 'test_model'
        mock_log3.value = 3.0
        mock_log4 = mocker.Mock()
        mock_log4.epoch = 2
        mock_log4.metric_name = 'test_model_2'
        mock_log4.value = 5.0
        mock_list = [mock_log3, mock_log4]
        mock_query = mocker.MagicMock()
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = mock_list.__iter__()
        self.mock_context.query.return_value = mock_query

        with pytest.raises(exceptions.TrackerError) as err:
            _ = tracker_started._get_run_metrics([], -1)

        assert err.match('test_model')
        assert err.match('test_model_2')

    def test_start_experiment_run_creation(
        self,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test StartExperimentEvent creates Run with the correct parameters."""
        tracker.notify(start_experiment_mock_event)
        self.Run.assert_called_once_with(
            start_experiment_mock_event.run_id,
            start_experiment_mock_event.run_ts,
            self.exp,
        )

    def test_start_experiment_tags_creation(
        self,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test that tags are created for each tag in the event."""
        start_experiment_mock_event.tags = ['tag1', 'tag2']
        tracker.notify(start_experiment_mock_event)
        assert self.Tags.call_count == len(start_experiment_mock_event.tags)

    def test_runner_registration_source_creation(
        self,
        tracker_started,
        runner_registration_mock_event,
    ) -> None:
        """Test RunnerRegistrationEvent creates Source correctly."""
        tracker_started.notify(runner_registration_mock_event)
        self.Source.assert_called_once_with(
            model_name=runner_registration_mock_event.model_name,
            model_ts=runner_registration_mock_event.model_ts,
            source_name=runner_registration_mock_event.runner_name,
            source_ts=runner_registration_mock_event.runner_ts,
            run=self.mock_context.merge.return_value,
        )
        self.TestSource.assert_not_called()

    def test_test_registration_source_creation(
        self,
        tracker_started,
        test_registration_mock_event,
    ) -> None:
        """Test TestRegistrationEvent creates Source and TestSource."""
        tracker_started.notify(test_registration_mock_event)
        self.Source.assert_called_once_with(
            model_name=test_registration_mock_event.model_name,
            model_ts=test_registration_mock_event.model_ts,
            source_name=test_registration_mock_event.runner_name,
            source_ts=test_registration_mock_event.runner_ts,
            run=self.mock_context.merge.return_value,
        )
        self.TestSource.assert_called_once_with(source=self.Source.return_value)

    def test_metric_event_log_creation(
        self,
        tracker_started,
        runner_registration_mock_event,
        epoch_metrics_mock_event,
    ) -> None:
        """Test that MetricEvent creates Log entries for each metric."""
        tracker_started.notify(runner_registration_mock_event)
        epoch_metrics_mock_event.metrics = {'metric1': 1.0, 'metric2': 2.0}
        tracker_started.notify(epoch_metrics_mock_event)
        assert self.Log.call_count == len(epoch_metrics_mock_event.metrics)

    def test_sqlalchemy_query_usage(
        self,
        mocker,
        tracker_started,
    ) -> None:
        """Test that SQLAlchemy queries use the correct syntax."""
        mock_query = mocker.MagicMock()
        mock_run = mocker.Mock()
        mock_run.run_id = 'test_run_id'
        mock_query.join.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = [].__iter__()
        self.mock_context.query.return_value = mock_query
        self.mock_context.merge.return_value = mock_run

        assert tracker_started._find_sources('test_model') == {}
        mock_query.where.assert_called_once()

    def test_load_metrics_method(
        self,
        mocker,
        tracker_started,
    ) -> None:
        """Test the _load_metrics method integration."""
        mock_sources = {'source1': [self.source]}
        mock_metrics = ([1, 2], {'metric1': [1.0, 2.0]})
        find_sources = mocker.patch.object(
            tracker_started, '_find_sources', return_value=mock_sources
        )
        get_run_metrics = mocker.patch.object(
            tracker_started, '_get_run_metrics', return_value=mock_metrics
        )
        result = tracker_started._load_metrics('test_model', max_epoch=10)

        assert 'source1' in result
        assert result['source1'] == mock_metrics
        find_sources.assert_called_once_with('test_model')
        get_run_metrics.assert_called_once_with([self.source], max_epoch=10)

    def test_load_test_metrics_method(
        self,
        mocker,
        tracker_started,
    ) -> None:
        """Test the _load_test_metrics method integration."""
        mock_sources = {'test_exp': {'test_source': [self.source]}}
        mock_metrics = ([1], {'metric1': [1.0]})
        find_test_sources = mocker.patch.object(
            tracker_started, '_find_test_sources', return_value=mock_sources
        )
        get_run_metrics = mocker.patch.object(
            tracker_started, '_get_run_metrics', return_value=mock_metrics
        )
        result = tracker_started._load_test_metrics('test_model')

        assert 'test_exp' in result
        assert 'test_source' in result['test_exp']
        assert result['test_exp']['test_source'] == mock_metrics
        find_test_sources.assert_called_once_with('test_model')
        get_run_metrics.assert_called_once_with([self.source], max_epoch=-1)

    def test_pause_stashes_and_continue_restores(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that pause stashes state and continue restores it."""
        tracker.notify(start_experiment_mock_event)
        original_run = tracker._run

        tracker.notify(pause_experiment_mock_event)
        assert tracker._run is None

        tracker.notify(continue_experiment_mock_event)
        assert tracker._run == original_run

    def test_pause_start_stop_continue_keeps_state(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        stop_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that concurrent runs do not corrupt stashed state."""
        tracker.notify(start_experiment_mock_event)
        original_run = tracker._run
        tracker.notify(pause_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.run_id = 'run2'

        tracker.notify(start_2)

        stop_2 = copy.copy(stop_experiment_mock_event)
        stop_2.run_id = 'run2'
        tracker.notify(stop_2)

        tracker.notify(continue_experiment_mock_event)
        assert tracker._run == original_run

    def test_continue_without_stash_raises_error(
        self, tracker, continue_experiment_mock_event
    ) -> None:
        """Test that continuing without a stash raises an error."""
        with pytest.raises(exceptions.NoStashedStateError):
            tracker.notify(continue_experiment_mock_event)

    def test_close_releases_stashes(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
    ) -> None:
        """Test that close unconditionally releases stashed resources."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)

        tracker.close()

        assert self.mock_engine.dispose.call_count == 1
        assert tracker._sql_stashed_runs == {}
        assert tracker._sql_stashed_sources == {}

    def test_start_experiment_reuses_existing_experiment(
        self,
        mocker,
        tracker,
        start_experiment_mock_event,
    ) -> None:
        """Test StartExperimentEvent reuses existing experiment and tags."""
        existing_tag = mocker.Mock(text='tag1')
        mock_where = self.mock_context.query.return_value.where.return_value
        mock_where.first.return_value = self.exp
        self.exp.tags = [existing_tag]
        start_experiment_mock_event.tags = ['tag1', 'tag2']

        tracker.notify(start_experiment_mock_event)

        self.Experiment.assert_not_called()
        self.Run.assert_called_once_with(
            start_experiment_mock_event.run_id,
            start_experiment_mock_event.run_ts,
            self.exp,
        )
        assert self.Tags.call_count == 1
        self.Tags.assert_called_once_with(text='tag2', experiment=self.exp)


class TestSQLConnectionInMemory:
    """Tests with a real in-memory engine."""

    def test_in_memory_engine_survives_between_runs(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Test that in-memory SQLite database survives between runs."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)
        start_experiment_mock_event.tags = ['tag1', 'tag2']
        try:
            start_event_2 = dataclasses.replace(
                start_experiment_mock_event,
                run_id='run2',
                tags=['tag2', 'tag3'],
            )
        except TypeError:
            start_event_2 = copy.copy(start_experiment_mock_event)
            start_event_2.run_id = 'run2'
            start_event_2.tags = ['tag2', 'tag3']

        tracker.notify(start_experiment_mock_event)
        tracker.notify(stop_experiment_mock_event)
        tracker.notify(start_event_2)

        with tracker.session_factory() as session:
            experiments = session.query(Experiment).all()
            runs = session.query(Run).all()
            tags = session.query(Tags).all()
            num_experiments = len(experiments)
            num_runs = len(runs)
            exp_runs_count = len(experiments[0].runs) if experiments else 0
            tag_texts = {t.text for t in tags}
            num_tags = len(tags)

        assert num_experiments == 1
        assert num_runs == 2
        assert exp_runs_count == 2
        assert tag_texts == {'tag1', 'tag2', 'tag3'}
        assert num_tags == 3

    def test_test_only_model_raises_no_sources(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """A test-only model raises NoSourcesError and loads test metrics."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)
        tracker.notify(start_experiment_mock_event)
        test_event = log_events.TestRegistrationEvent(
            runner_name='test_runner',
            runner_ts=start_experiment_mock_event.run_ts,
            model_name='tested_only_model',
            model_ts=start_experiment_mock_event.run_ts,
            metadata='',
        )
        metric_event = log_events.TestMetricEvent(
            model_name='tested_only_model',
            source_name='test_runner',
            epoch=1,
            metrics={'accuracy': 0.95},
        )
        tracker.notify(test_event)
        tracker.notify(metric_event)

        with pytest.raises(exceptions.NoSourcesError):
            _ = tracker.load_metrics('tested_only_model')

        test_metrics = tracker.load_test_metrics('tested_only_model')
        tracker.notify(stop_experiment_mock_event)

        assert test_metrics == {
            'test_exp': {'test_runner': ([1], {'accuracy': [0.95]})},
        }
        return

    def test_load_test_metrics_multiple_experiments_last_run(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """load_test_metrics returns the last run of each experiment."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'exp_1'
        start_1.run_id = 'run_1'
        tracker.notify(start_1)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_old',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_old',
                epoch=1,
                metrics={'acc': 0.8},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.exp_name = 'exp_1'
        start_2.run_id = 'run_2'
        tracker.notify(start_2)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_new',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_new',
                epoch=2,
                metrics={'acc': 0.9},
            )
        )
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_other',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_b',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_b',
                source_name='test_other',
                epoch=2,
                metrics={'acc': 0.7},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        start_3 = copy.copy(start_experiment_mock_event)
        start_3.exp_name = 'exp_2'
        start_3.run_id = 'run_1'
        tracker.notify(start_3)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_exp2',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_exp2',
                epoch=1,
                metrics={'acc': 0.85},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        results = tracker.load_test_metrics('model_a')
        assert set(results) == {'exp_1', 'exp_2'}
        assert set(results['exp_1']) == {'test_new'}
        assert set(results['exp_2']) == {'test_exp2'}
        return

    def test_load_test_metrics_resumed_run(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Resumed runs return all tests of the run, and later run replaces."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'exp'
        start_1.run_id = 'run_1'
        tracker.notify(start_1)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_early',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_early',
                epoch=1,
                metrics={'acc': 0.7},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.exp_name = 'exp'
        start_2.run_id = 'run_2'
        tracker.notify(start_2)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_part_1',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_part_1',
                epoch=1,
                metrics={'acc': 0.8},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        continue_2 = copy.copy(start_experiment_mock_event)
        continue_2.exp_name = 'exp'
        continue_2.run_id = 'run_2'
        tracker.notify(continue_2)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_part_2',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_part_2',
                epoch=1,
                metrics={'acc': 0.9},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        results = tracker.load_test_metrics('model_a')
        assert set(results) == {'exp'}
        assert set(results['exp']) == {'test_part_1', 'test_part_2'}
        assert 'test_early' not in results['exp']

    def test_rename_experiment(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Rename experiment, keep runs and tags, and update test metrics."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'old_name'
        start_1.run_id = 'run_1'
        start_1.tags = ['tag1', 'tag2']
        tracker.notify(start_1)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_s1',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_s1',
                epoch=1,
                metrics={'acc': 0.8},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.exp_name = 'old_name'
        start_2.run_id = 'run_2'
        start_2.tags = ['tag2', 'tag3']
        tracker.notify(start_2)
        tracker.notify(stop_experiment_mock_event)

        tracker.rename_experiment('old_name', 'new_name')

        results = tracker.load_test_metrics('model_a')
        with tracker.session_factory() as session:
            old_exp = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('old_name'))
                .first()
            )
            new_exp = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('new_name'))
                .first()
            )
            runs = [r.run_id for r in new_exp.runs] if new_exp else []
            tags = {t.text for t in new_exp.tags} if new_exp else set()

        assert 'new_name' in results
        assert 'old_name' not in results
        assert old_exp is None
        assert new_exp is not None
        assert set(runs) == {'run_1', 'run_2'}
        assert tags == {'tag1', 'tag2', 'tag3'}

    def test_delete_experiment(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Delete an experiment and all its cascaded and uncascaded rows."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'exp_1'
        start_1.run_id = 'run_1'
        start_1.tags = ['tag1']
        tracker.notify(start_1)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_runner_1',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_runner_1',
                epoch=1,
                metrics={'acc': 0.8},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.exp_name = 'exp_2'
        start_2.run_id = 'run_2'
        start_2.tags = ['tag2']
        tracker.notify(start_2)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_runner_2',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_b',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_b',
                source_name='test_runner_2',
                epoch=1,
                metrics={'acc': 0.9},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        tracker.delete_experiment('exp_1')

        with tracker.session_factory() as session:
            exp_1 = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('exp_1'))
                .first()
            )
            exp_2 = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('exp_2'))
                .first()
            )
            n_experiments = session.query(Experiment).count()
            n_tags = session.query(Tags).count()
            n_runs = session.query(Run).count()
            n_sources = session.query(Source).count()
            n_test_sources = session.query(TestSource).count()
            n_logs = session.query(Log).count()

        assert exp_1 is None
        assert exp_2 is not None
        assert n_experiments == 1
        assert n_tags == 1
        assert n_runs == 1
        assert n_sources == 1
        assert n_test_sources == 1
        assert n_logs == 1

    def test_delete_experiment_unknown_name_raises(self) -> None:
        """Deleting a nonexistent experiment raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        with pytest.raises(
            exceptions.TrackerError, match='No experiment named'
        ):
            tracker.delete_experiment('nonexistent')

    def test_delete_experiment_active_run_raises(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Deleting an experiment while a run is active raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)
        tracker.notify(start_experiment_mock_event)

        with pytest.raises(exceptions.TrackerError, match='active or stashed'):
            tracker.delete_experiment(start_experiment_mock_event.exp_name)

        tracker.notify(stop_experiment_mock_event)

    def test_rename_experiment_unknown_name_raises(self) -> None:
        """Renaming a nonexistent experiment raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        with pytest.raises(
            exceptions.TrackerError, match='No experiment named'
        ):
            tracker.rename_experiment('nonexistent', 'new_name')

    def test_rename_experiment_existing_new_name_raises(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Renaming onto an existing experiment name raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'exp_1'
        tracker.notify(start_1)
        tracker.notify(stop_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.exp_name = 'exp_2'
        tracker.notify(start_2)
        tracker.notify(stop_experiment_mock_event)

        with pytest.raises(exceptions.TrackerError, match='already exists'):
            tracker.rename_experiment('exp_1', 'exp_2')

    def test_rename_experiment_active_run_raises(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Renaming an experiment while a run is active raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)
        tracker.notify(start_experiment_mock_event)

        with pytest.raises(exceptions.TrackerError, match='active or stashed'):
            tracker.rename_experiment(
                start_experiment_mock_event.exp_name, 'new_name'
            )

        tracker.notify(stop_experiment_mock_event)

    def test_clone_experiment(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Clone experiment keeps runs, tags, test sources and logs."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'exp_1'
        start_1.run_id = 'run_1'
        start_1.tags = ['tag1', 'tag2']
        tracker.notify(start_1)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_s1',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_s1',
                epoch=1,
                metrics={'acc': 0.8},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        start_resumed = copy.copy(start_experiment_mock_event)
        start_resumed.exp_name = 'exp_1'
        start_resumed.run_id = 'run_1'
        start_resumed.resumed = True
        start_resumed.tags = []
        tracker.notify(start_resumed)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_s1',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_s1',
                epoch=2,
                metrics={'acc': 0.9},
            )
        )
        tracker.notify(stop_experiment_mock_event)

        tracker.clone_experiment('exp_1', 'exp_clone')

        results = tracker.load_test_metrics('model_a')
        with tracker.session_factory() as session:
            clone_exp = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('exp_clone'))
                .first()
            )
            clone_runs = [r.run_id for r in clone_exp.runs] if clone_exp else []
            clone_tags = (
                {t.text for t in clone_exp.tags} if clone_exp else set()
            )

        assert results['exp_1'] == results['exp_clone']
        assert clone_runs == ['run_1', 'run_1']
        assert clone_tags == {'tag1', 'tag2'}

    def test_clone_experiment_modification_leaves_original_untouched(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Modifying or deleting a clone leaves the original untouched."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'exp_1'
        start_1.run_id = 'run_1'
        tracker.notify(start_1)
        tracker.notify(
            log_events.TestRegistrationEvent(
                runner_name='test_s1',
                runner_ts=start_experiment_mock_event.run_ts,
                model_name='model_a',
                model_ts=start_experiment_mock_event.run_ts,
                metadata='',
            )
        )
        tracker.notify(
            log_events.TestMetricEvent(
                model_name='model_a',
                source_name='test_s1',
                epoch=1,
                metrics={'acc': 0.8},
            )
        )
        tracker.notify(stop_experiment_mock_event)
        tracker.clone_experiment('exp_1', 'exp_clone')

        tracker.rename_experiment('exp_clone', 'exp_clone_renamed')
        tracker.delete_experiment('exp_clone_renamed')

        results = tracker.load_test_metrics('model_a')
        with tracker.session_factory() as session:
            orig_exp = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('exp_1'))
                .first()
            )
            clone_exp = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('exp_clone'))
                .first()
            )
            renamed_exp = (
                session.query(Experiment)
                .where(Experiment.experiment_name.is_('exp_clone_renamed'))
                .first()
            )

        assert 'exp_1' in results
        assert 'exp_clone' not in results
        assert 'exp_clone_renamed' not in results
        assert orig_exp is not None
        assert clone_exp is None
        assert renamed_exp is None

    def test_clone_experiment_unknown_name_raises(self) -> None:
        """Cloning a nonexistent experiment raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        with pytest.raises(
            exceptions.TrackerError, match='No experiment named'
        ):
            tracker.clone_experiment('nonexistent', 'new_name')

    def test_clone_experiment_existing_new_name_raises(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Cloning onto an existing experiment name raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)

        start_1 = copy.copy(start_experiment_mock_event)
        start_1.exp_name = 'exp_1'
        tracker.notify(start_1)
        tracker.notify(stop_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.exp_name = 'exp_2'
        tracker.notify(start_2)
        tracker.notify(stop_experiment_mock_event)

        with pytest.raises(exceptions.TrackerError, match='already exists'):
            tracker.clone_experiment('exp_1', 'exp_2')

    def test_clone_experiment_active_run_raises(
        self,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> None:
        """Cloning an experiment while a run is active raises TrackerError."""
        engine = sqlalchemy.create_engine('sqlite://')
        tracker = SQLConnection(engine=engine)
        tracker.notify(start_experiment_mock_event)

        with pytest.raises(exceptions.TrackerError, match='active or stashed'):
            tracker.clone_experiment(
                start_experiment_mock_event.exp_name, 'new_name'
            )

        tracker.notify(stop_experiment_mock_event)

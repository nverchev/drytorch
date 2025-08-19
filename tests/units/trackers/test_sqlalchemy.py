"""Tests for the "sqlalchemy" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('sqlalchemy'):
    pytest.skip('sqlalchemy not available', allow_module_level=True)

from collections.abc import Generator

from drytorch.core import exceptions
from drytorch.trackers.sqlalchemy import (
    Experiment,
    Log,
    Run,
    Source,
    SQLConnection,
    Tags,
)


class TestSQLConnection:
    """Tests for the SQLConnection tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Setup test environment."""
        self.mock_engine = 'mock_engine'
        self.mock_context = mocker.Mock()
        self.mock_session = mocker.MagicMock()
        self.mock_session.__enter__.return_value = self.mock_context
        self.MockSession = mocker.Mock(return_value=self.mock_session)
        self.create_engine_mock = mocker.Mock(return_value=self.mock_engine)
        self.make_mock_session = mocker.Mock(return_value=self.MockSession)
        self.exp = mocker.create_autospec(Experiment, instance=True)
        self.log = mocker.create_autospec(Log, instance=True)
        self.run = mocker.create_autospec(Run, instance=True)
        self.source = mocker.create_autospec(Source, instance=True)
        self.tags = mocker.create_autospec(Tags, instance=True)
        self.mock_last_run = mocker.Mock()
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
        return

    @pytest.fixture
    def tracker(self) -> SQLConnection:
        """Set up the instance."""
        return SQLConnection()

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

    def test_cleanup(self, tracker_started):
        """Test correct cleaning up."""
        tracker_started.clean_up()
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
        assert tracker_started.run == self.run
        self.mock_context.add.assert_called_with(self.exp)

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
        actor_registration_mock_event,
    ) -> None:
        """Test call model notification creates the source."""
        tracker_started.notify(actor_registration_mock_event)
        self.mock_context.add.assert_called_with(self.source)
        sources = tracker_started._sources
        assert actor_registration_mock_event.actor_name in sources

    def test_notify_metrics(
        self,
        tracker_started,
        actor_registration_mock_event,
        epoch_metrics_mock_event,
    ) -> None:
        """Test metrics notification creates log entries."""
        tracker_started.notify(actor_registration_mock_event)
        tracker_started.notify(epoch_metrics_mock_event)
        self.mock_context.merge.assert_called_with(self.source)
        self.mock_context.add.assert_called_with(self.log)

    def test_unknown_source(
        self,
        tracker_started,
        actor_registration_mock_event,
        epoch_metrics_mock_event,
    ) -> None:
        """Test metrics notification from an unknown source raises an error."""
        tracker_started.notify(actor_registration_mock_event)
        epoch_metrics_mock_event.source_name = 'unknown_source'
        with pytest.raises(exceptions.TrackerError):
            tracker_started.notify(epoch_metrics_mock_event)

    def test_find_sources_nonexistent_model(
        self, mocker, tracker_started
    ) -> None:
        """Test _find_sources with a nonexistent model raises an exception."""
        mock_query = mocker.MagicMock()
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = [].__iter__()
        self.mock_context.query.return_value = mock_query
        with pytest.raises(exceptions.TrackerError):
            tracker_started._find_sources('nonexistent_model')

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

    def test_actor_registration_source_creation(
        self,
        tracker_started,
        actor_registration_mock_event,
    ) -> None:
        """Test ActorRegistrationEvent creates Source correctly."""
        tracker_started.notify(actor_registration_mock_event)
        self.Source.assert_called_once_with(
            model_name=actor_registration_mock_event.model_name,
            model_ts=actor_registration_mock_event.model_ts,
            source_name=actor_registration_mock_event.actor_name,
            source_ts=actor_registration_mock_event.actor_ts,
            run=self.mock_context.merge.return_value,
        )

    def test_metric_event_log_creation(
        self,
        tracker_started,
        actor_registration_mock_event,
        epoch_metrics_mock_event,
    ) -> None:
        """Test that MetricEvent creates Log entries for each metric."""
        tracker_started.notify(actor_registration_mock_event)
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
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = [].__iter__()
        self.mock_context.query.return_value = mock_query
        self.mock_context.merge.return_value = mock_run
        with pytest.raises(exceptions.TrackerError):
            tracker_started._find_sources('test_model')

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

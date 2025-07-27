"""Tests for the "sqlalchemy" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('sqlalchemy'):
    pytest.skip('sqlalchemy not available', allow_module_level=True)

import warnings

from collections.abc import Generator

from drytorch import exceptions
from drytorch.trackers.sqlalchemy import (
    Experiment,
    Log,
    Run,
    Source,
    SQLConnection,
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
        mocker.patch('sqlalchemy.create_engine', self.create_engine_mock)
        mocker.patch('sqlalchemy.orm.sessionmaker', self.make_mock_session)
        mocker.patch('sqlalchemy.schema.MetaData.create_all')
        mocker.patch('drytorch.trackers.sqlalchemy.Experiment',
                     return_value=self.exp)
        mocker.patch('drytorch.trackers.sqlalchemy.Log',
                     return_value=self.log)
        mocker.patch('drytorch.trackers.sqlalchemy.Run',
                     return_value=self.run)
        mocker.patch('drytorch.trackers.sqlalchemy.Source',
                     return_value=self.source)
        return

    @pytest.fixture
    def tracker(self) -> SQLConnection:
        """Set up the instance."""
        return SQLConnection()

    @pytest.fixture
    def tracker_with_resume(self) -> SQLConnection:
        """Set up the instance with resume."""
        return SQLConnection(resume_run=True)

    @pytest.fixture
    def tracker_started(
            self,
            tracker,
            start_experiment_mock_event,
            stop_experiment_mock_event,
    ) -> Generator[SQLConnection, None, None]:
        """Set up the instance with resume."""
        tracker.notify(start_experiment_mock_event)
        yield tracker

        tracker.notify(stop_experiment_mock_event)
        return

    def test_cleanup(self, tracker_started):
        """Test correct clean up."""
        tracker_started.clean_up()
        assert tracker_started._run is None
        assert tracker_started._sources == {}

    def test_init_default(self, tracker) -> None:
        """Test initialization with default parameters."""
        self.create_engine_mock.assert_called_once_with(tracker.default_url)
        self.make_mock_session.assert_called_once_with(bind=self.mock_engine)
        assert tracker.resume_run is False

    def test_init_with_resume(self, tracker_with_resume) -> None:
        """Test initialization with resume_run=True."""
        assert tracker_with_resume.resume_run is True

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
        self.mock_context.add.assert_called_once_with(self.exp)

    def test_notify_start_stop_experiment(
            self,
            tracker_started,
            stop_experiment_mock_event,
    ) -> None:
        """Test stop experiment notification cleans up the state."""
        tracker_started.notify(stop_experiment_mock_event)
        assert tracker_started._run is None

    def test_notify_start_experiment_with_resume_no_previous_run(
            self,
            mocker,
            tracker_with_resume,
            start_experiment_mock_event,
    ) -> None:
        """Test start experiment with resume when no previous run exists."""
        last_run = None
        get_last_run = mocker.patch.object(tracker_with_resume, '_get_last_run')
        get_last_run.return_value = last_run
        with warnings.catch_warnings(record=True) as w:
            tracker_with_resume.notify(start_experiment_mock_event)
            assert 'No previous runs' in str(w[0].message)
        run = tracker_with_resume.run
        assert run == self.run

    def test_notify_start_experiment_with_resume_existing_run(
            self,
            mocker,
            tracker_with_resume,
            start_experiment_mock_event,
    ) -> None:
        """Test start experiment with resume when the previous run exists."""
        # Create a previous run
        last_run = mocker.Mock()
        get_last_run = mocker.patch.object(tracker_with_resume, '_get_last_run')
        get_last_run.return_value = last_run
        tracker_with_resume.notify(start_experiment_mock_event)
        resumed_run = tracker_with_resume.run
        assert resumed_run == last_run

    def test_notify_call_model(
            self,
            tracker_started,
            call_model_mock_event,
    ) -> None:
        """Test call model notification creates the source."""
        tracker_started.notify(call_model_mock_event)
        self.mock_context.add.assert_called_with(self.source)
        assert call_model_mock_event.source_name in tracker_started._sources

    def test_notify_metrics(
            self,
            tracker_started,
            call_model_mock_event,
            epoch_metrics_mock_event,
    ) -> None:
        """Test metrics notification creates log entries."""
        tracker_started.notify(call_model_mock_event)
        tracker_started.notify(epoch_metrics_mock_event)
        self.mock_context.merge.assert_called_with(self.source)
        self.mock_context.add.assert_called_with(self.log)

    def test_unknown_source(
            self,
            tracker_started,
            call_model_mock_event,
            epoch_metrics_mock_event,
    ) -> None:
        """Test metrics notification from an unknown source raises an error."""
        tracker_started.notify(call_model_mock_event)
        epoch_metrics_mock_event.source_name = 'unknown_source'
        with pytest.raises(exceptions.TrackerError):
            tracker_started.notify(epoch_metrics_mock_event)

    def test_find_sources_existing_model(
            self,
            mocker,
            tracker_started,
    ) -> None:
        """Test _find_sources with an existing model."""
        self.source.source_name = 'test_source'
        mock_query = mocker.MagicMock()
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = [self.source].__iter__()
        self.mock_context.query.return_value = mock_query
        assert 'test_source' in tracker_started._find_sources('test_model')

    def test_find_sources_nonexistent_model(self,
                                            mocker,
                                            tracker_started) -> None:
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
        mock_log.value = 2.
        mock_log2 = mocker.Mock()
        mock_log2.epoch = 1
        mock_log2.metric_name = 'test_model_2'
        mock_log2.value = 4.
        mock_log3 = mocker.Mock()
        mock_log3.epoch = 2
        mock_log3.metric_name = 'test_model'
        mock_log3.value = 3.
        mock_log4 = mocker.Mock()
        mock_log4.epoch = 2
        mock_log4.metric_name = 'test_model_2'
        mock_log4.value = 5.

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
        mock_log3.value = 3.
        mock_log4 = mocker.Mock()
        mock_log4.epoch = 2
        mock_log4.metric_name = 'test_model_2'
        mock_log4.value = 5.
        mock_list = [mock_log3, mock_log4]
        mock_query = mocker.MagicMock()
        mock_query.where.return_value = mock_query
        mock_query.__iter__.return_value = mock_list.__iter__()
        self.mock_context.query.return_value = mock_query
        with pytest.raises(exceptions.TrackerError) as err:
            _ = tracker_started._get_run_metrics([], -1)
        assert err.match('test_model')
        assert err.match('test_model_2')

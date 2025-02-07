"""Tests for the tracking module."""
from typing import Optional

import pytest

import functools

import dry_torch
from dry_torch import exceptions
from dry_torch import log_events
from dry_torch.tracking import EventDispatcher, Experiment, MetadataManager
from dry_torch.tracking import Tracker
from tests.units.conftest import experiment_current_original


@pytest.fixture(autouse=True, scope='module')
def remove_trackers() -> None:
    """Remove trackers."""
    dry_torch.remove_all_default_trackers()
    return


class _SimpleEvent(log_events.Event):
    """Simple Event subclass for testing."""
    pass


class _UndefinedEvent(log_events.Event):
    """Event subclass that is not handled by tracker."""
    pass


class _SimpleTracker(Tracker):
    """Simple tracker that saves the last event."""
    last_event: Optional[log_events.Event] = None

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        pass

    @notify.register
    def _(self, event: _SimpleEvent) -> None:
        self.last_event = event

    @notify.register
    def _(self, event: _UndefinedEvent) -> None:
        raise NotImplementedError('`notify` is not implemented.')


# Test class for MetadataManager
class TestMetadataManager:
    """Test MetadataManager functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the MetadataManager for testing."""
        self.max_items_repr = 10
        self.manager = MetadataManager(max_items_repr=self.max_items_repr)

    def test_record_model_call(self, mocker) -> None:
        """Test recording metadata creates the event."""
        mock_obj = mocker.Mock()
        mock_model_name = 'test_model'
        mock_name = 'test_caller'

        mock_log_event = mocker.patch('dry_torch.log_events.CallModel')
        self.manager.record_model_call(mock_name, mock_model_name, mock_obj)
        assert mock_name in self.manager.used_names
        mock_log_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            self.manager.record_model_call(mock_name, mock_model_name, mock_obj)

    def test_register_model(self, mocker, mock_model) -> None:
        """Test registering a model creates the event."""

        mock_log_event = mocker.patch('dry_torch.log_events.ModelCreation')
        self.manager.register_model(mock_model)
        assert mock_model.name in self.manager.used_names
        mock_log_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            self.manager.register_model(mock_model)

    def test_extract_metadata(self, mocker) -> None:
        """Test metadata extraction with a recursive_repr wrapper."""
        mock_obj = mocker.Mock()

        mocker.patch('dry_torch.repr_utils.recursive_repr',
                     return_value={'key': 'value'})

        metadata = self.manager.extract_metadata(mock_obj, max_size=5)
        assert metadata == {'key': 'value'}

    def test_extract_metadata_recursion_error(self, mocker) -> None:
        """Test extract_metadata handles RecursionError gracefully."""
        mock_obj = mocker.Mock()

        mocker.patch('dry_torch.repr_utils.recursive_repr',
                     side_effect=RecursionError)
        with pytest.warns(exceptions.RecursionWarning):
            _ = self.manager.extract_metadata(mock_obj, max_size=5)


class TestEventDispatcher:
    """Test the event dispatcher."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path) -> None:
        """Set up an experiment."""
        self.name = 'TestExperiment'
        self.par_dir = tmp_path
        self.dispatcher = EventDispatcher(self.name)
        self.tracker = _SimpleTracker()
        self.dispatcher.register(self.tracker)
        return

    def test_register_tracker(self):
        """Test that a tracker can be registered to an experiment."""
        assert self.tracker.__class__.__name__ in self.dispatcher.named_trackers

    def test_register_duplicate_tracker_raises_error(self):
        """Test that registering a duplicate tracker raises an error."""
        with pytest.raises(exceptions.TrackerAlreadyRegisteredError):
            self.dispatcher.register(_SimpleTracker())

    def test_remove_named_tracker(self):
        """Test that a registered tracker can be removed by name."""
        tracker_name = self.tracker.__class__.__name__
        self.dispatcher.remove(tracker_name)
        assert tracker_name not in self.dispatcher.named_trackers

    def test_remove_nonexistent_tracker_raises_error(self):
        """Test that removing a non-existent tracker raises an error."""
        with pytest.raises(exceptions.TrackerNotRegisteredError):
            self.dispatcher.remove('NonexistentTracker')

    def test_publish_event_calls_tracker_notify(self):
        """Test publishing an event calls notify on registered trackers."""
        simple_event = _SimpleEvent()
        self.dispatcher.publish(event=simple_event)
        assert self.tracker.last_event is simple_event

    def test_handle_tracker_exceptions(self):
        """Test handling of tracker exceptions."""
        simple_event = _UndefinedEvent()
        with pytest.warns(exceptions.TrackerError):
            self.dispatcher.publish(event=simple_event)


class TestExperiment:
    """Test the Experiment class."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path) -> None:
        """Set up an experiment."""
        self.name = 'TestExperiment'
        self.par_dir = tmp_path
        self.experiment = Experiment[None](self.name, self.par_dir)
        setattr(self.experiment, 'current', experiment_current_original)
        setattr(Experiment,
                'current',
                experiment_current_original)
        setattr(self.experiment.__class__,
                'current',
                experiment_current_original)
        return

    def test_start_and_stop_experiment(self, mocker):
        """Test starting and stopping an experiment."""
        mock_event_start = mocker.patch.object(log_events, 'StartExperiment')
        mock_event_stop = mocker.patch.object(log_events, 'StopExperiment')
        with self.experiment:
            mock_event_start.assert_called_once_with(self.experiment.name,
                                                     self.par_dir / self.name)
        mock_event_stop.assert_called_once_with(self.experiment.name)

    def test_get_config_no_config_error(self):
        """Test NoConfigError is raised if config is None."""
        with self.experiment:
            with pytest.raises(exceptions.NoConfigError):
                Experiment.get_config()

    def test_no_active_experiment_error(self):
        """Test that error is called when no experiment is active."""
        with pytest.raises(exceptions.NoActiveExperimentError):
            # Experiment.current has been stored in experiment_current_original
            _ = self.experiment.current()

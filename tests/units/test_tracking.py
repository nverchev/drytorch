"""Tests for the "tracking" module."""

import dataclasses
from typing import Optional

import pytest

import functools

from drytorch import exceptions
from drytorch import log_events
from drytorch.tracking import EventDispatcher
from drytorch.tracking import MetadataManager
from drytorch.tracking import Tracker
from drytorch.tracking import remove_all_default_trackers


@pytest.fixture(autouse=True, scope='module')
def remove_trackers() -> None:
    """Remove trackers."""
    remove_all_default_trackers()
    return


@dataclasses.dataclass
class _SimpleEvent(log_events.Event):
    """Simple Event subclass for testing."""
    pass


@dataclasses.dataclass
class _UndefinedEvent(log_events.Event):
    """Event subclass that the tracker does not handle."""
    pass


class TestEvent:

    def test_no_auto_publish(self):
        """Test the error raises correctly when instantiating the class."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _SimpleEvent()


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

    def test_record_model_call(self, mocker, mock_model) -> None:
        """Test recording metadata creates the event."""
        mock_obj = mocker.Mock()
        mock_obj.name = 'mock obj'
        mock_log_event = mocker.patch('drytorch.log_events.CallModel')
        self.manager.record_model_call(mock_obj, mock_model)
        assert mock_obj.name in self.manager.used_names
        mock_log_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            self.manager.record_model_call(mock_obj, mock_model)

    def test_register_model(self, mocker, mock_model) -> None:
        """Test registering a model creates the event."""

        mock_log_event = mocker.patch('drytorch.log_events.ModelCreation')
        self.manager.register_model(mock_model)
        assert mock_model.name in self.manager.used_names
        mock_log_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            self.manager.register_model(mock_model)

    def test_extract_metadata(self, mocker) -> None:
        """Test metadata extraction with a recursive_repr wrapper."""
        mock_obj = mocker.Mock()

        mocker.patch('drytorch.utils.repr_utils.recursive_repr',
                     return_value={'key': 'value'})

        metadata = self.manager.extract_metadata(mock_obj, max_size=5)
        assert metadata == {'key': 'value'}

    def test_extract_metadata_recursion_error(self, mocker) -> None:
        """Test extract_metadata handles RecursionError gracefully."""
        mock_obj = mocker.Mock()

        mocker.patch('drytorch.utils.repr_utils.recursive_repr',
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
        """Test that a registered tracker can be removed by its name."""
        tracker_name = self.tracker.__class__.__name__
        self.dispatcher.remove(tracker_name)
        assert tracker_name not in self.dispatcher.named_trackers

    def test_remove_nonexistent_tracker_raises_error(self):
        """Test that removing a non-existent tracker raises an error."""
        with pytest.raises(exceptions.TrackerNotRegisteredError):
            self.dispatcher.remove('NonexistentTracker')

    def test_publish_event_calls_tracker_notify(self):
        """Test publishing an event notifies registered trackers."""
        _SimpleEvent.set_auto_publish(self.dispatcher.publish)
        simple_event = _SimpleEvent()
        assert self.tracker.last_event is simple_event

    def test_handle_tracker_exceptions(self):
        """Test handling of tracker exceptions."""
        _UndefinedEvent.set_auto_publish(self.dispatcher.publish)
        with pytest.warns(exceptions.TrackerError):
            _UndefinedEvent()

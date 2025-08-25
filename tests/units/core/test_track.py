"""Tests for the "track" module."""

import dataclasses
import functools

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.track import (
    EventDispatcher,
    MetadataManager,
    Tracker,
    remove_all_default_trackers,
)


@pytest.fixture(autouse=True, scope='module')
def remove_trackers() -> None:
    """Remove trackers."""
    remove_all_default_trackers()
    return


@dataclasses.dataclass(frozen=True)
class _SimpleEvent(log_events.Event):
    """Simple Event subclass for testing."""


@dataclasses.dataclass(frozen=True)
class _UndefinedEvent(log_events.Event):
    """Event subclass that the tracker does not handle."""


class _SimpleActor:
    """Simple actor for testing."""

    name: str = 'mock obj'


class TestEvent:
    """Tests for Event."""

    def test_no_auto_publish(self):
        """Test the error raises correctly when instantiating the class."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _SimpleEvent()


class _SimpleTracker(Tracker):
    """Simple tracker that saves the last event."""

    last_event: log_events.Event | None = None

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
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_model_registration_event = mocker.patch(
            'drytorch.core.log_events.ModelRegistrationEvent'
        )
        self.mock_caller_registration_event = mocker.patch(
            'drytorch.core.log_events.ActorRegistrationEvent'
        )
        return

    @pytest.fixture()
    def simple_actor(self) -> _SimpleActor:
        """Set up an actor instance for testing."""
        return _SimpleActor()

    @pytest.fixture()
    def manager(self) -> MetadataManager:
        """Set up the MetadataManager for testing."""
        return MetadataManager()

    def test_register_actor(self, mock_model, simple_actor, manager) -> None:
        """Test recording metadata creates the event."""
        manager.register_actor(simple_actor, mock_model)
        assert simple_actor.name in manager.metadata_dict
        self.mock_caller_registration_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            manager.register_actor(simple_actor, mock_model)

    def test_register_model(self, mock_model, manager) -> None:
        """Test registering a model creates the event."""
        manager.register_model(mock_model)
        assert mock_model.name in manager.metadata_dict
        self.mock_model_registration_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            manager.register_model(mock_model)

    def test_unregister_actor(self, simple_actor, mock_model, manager) -> None:
        """Test unregistering an actor removes its name from the used names."""
        manager.register_actor(simple_actor, mock_model)
        manager.unregister_actor(simple_actor)
        assert simple_actor.name not in manager.metadata_dict

    def test_unregister_model(self, mock_model, manager) -> None:
        """Test unregistering a model removes its name from the used names."""
        manager.register_model(mock_model)
        manager.unregister_actor(mock_model)
        assert mock_model.name not in manager.metadata_dict

    def test_extract_metadata_recursion_error(self, mocker, manager) -> None:
        """Test extract_metadata handles RecursionError gracefully."""
        mock_obj = mocker.Mock()

        mocker.patch(
            'drytorch.utils.repr_utils.recursive_repr',
            side_effect=RecursionError,
        )
        with pytest.warns(exceptions.RecursionWarning):
            _ = manager.extract_metadata(mock_obj)


class TestEventDispatcher:
    """Test the event dispatcher."""

    @pytest.fixture()
    def tracker(self) -> _SimpleTracker:
        """Set up a test instance."""
        return _SimpleTracker()

    @pytest.fixture()
    def dispatcher(self, tracker) -> EventDispatcher:
        """Set up a test instance."""
        dispatcher = EventDispatcher('TestExperiment')
        dispatcher.register(tracker)
        return dispatcher

    def test_register_tracker(self, tracker, dispatcher):
        """Test that a tracker can be registered to an experiment."""
        assert tracker.__class__.__name__ in dispatcher.named_trackers

    def test_register_duplicate_tracker_raises_error(self, dispatcher):
        """Test that registering a duplicate tracker raises an error."""
        with pytest.raises(exceptions.TrackerAlreadyRegisteredError):
            dispatcher.register(_SimpleTracker())

    def test_remove_named_tracker(self, tracker, dispatcher):
        """Test that a registered tracker can be removed by its name."""
        tracker_name = tracker.__class__.__name__
        dispatcher.remove(tracker_name)
        assert tracker_name not in dispatcher.named_trackers

    def test_remove_nonexistent_tracker_raises_error(self, dispatcher):
        """Test that removing a non-existent tracker raises an error."""
        with pytest.raises(exceptions.TrackerNotActiveError):
            dispatcher.remove('NonexistentTracker')

    def test_publish_event_calls_tracker_notify(self, tracker, dispatcher):
        """Test publishing an event notifies registered trackers."""
        _SimpleEvent.set_auto_publish(dispatcher.publish)
        simple_event = _SimpleEvent()
        assert tracker.last_event is simple_event

    def test_handle_tracker_exceptions(self, dispatcher):
        """Test handling of tracker exceptions."""
        _UndefinedEvent.set_auto_publish(dispatcher.publish)
        with pytest.warns(exceptions.TrackerExceptionWarning):
            _UndefinedEvent()

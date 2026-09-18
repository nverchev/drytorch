"""Tests for the ``tracking`` module."""

import dataclasses
import functools

from collections.abc import Generator

import pytest

from drytorch.core import exceptions, log_events
from drytorch.core.tracking import (
    DEFAULT_TRACKERS,
    EventDispatcher,
    MetadataManager,
    Tracker,
    extend_default_trackers,
    remove_all_default_trackers,
)


@pytest.fixture(autouse=True, scope='module')
def remove_trackers() -> Generator[None, None, None]:
    """Remove trackers."""
    remove_all_default_trackers()
    yield
    log_events.Event.set_auto_publish(None)
    return


class _SimpleRunner:
    """Simple runner for testing."""

    name: str = 'mock obj'


@dataclasses.dataclass(frozen=True)
class _SimpleEvent(log_events.Event):
    """Simple Event subclass for testing."""


@dataclasses.dataclass(frozen=True)
class _UndefinedEvent(log_events.Event):
    """Event subclass that the tracker does not handle."""


class _SimpleTracker(Tracker):
    """Simple tracker that saves the last event."""

    last_event: log_events.Event | None = None

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: _SimpleEvent) -> None:
        self.last_event = event
        return super().notify(event)

    @notify.register
    def _(self, event: _UndefinedEvent) -> None:
        raise NotImplementedError('`notify` is not implemented.')


class _ExitTracker(Tracker):
    """Tracker that raises SystemExit on event."""

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: _SimpleEvent) -> None:
        raise SystemExit()


class TestMetadataManager:
    """Test MetadataManager functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_model_registration_event = mocker.patch(
            'drytorch.core.log_events.ModelRegistrationEvent'
        )
        self.mock_caller_registration_event = mocker.patch(
            'drytorch.core.log_events.RunnerRegistrationEvent'
        )
        return

    @pytest.fixture()
    def simple_runner(self) -> _SimpleRunner:
        """Set up a runner instance for testing."""
        return _SimpleRunner()

    @pytest.fixture()
    def manager(self) -> MetadataManager:
        """Set up the MetadataManager for testing."""
        return MetadataManager()

    def test_register_runner(self, mock_model, simple_runner, manager) -> None:
        """Test recording metadata creates the event."""
        manager.register_runner(simple_runner, mock_model)

        assert simple_runner.name in manager.named_metadata
        self.mock_caller_registration_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            manager.register_runner(simple_runner, mock_model)

    def test_register_model(self, mock_model, manager) -> None:
        """Test registering a model creates the event."""
        manager.register_model(mock_model)

        assert mock_model.name in manager.named_metadata
        self.mock_model_registration_event.assert_called_once()
        with pytest.raises(exceptions.NameAlreadyRegisteredError):
            manager.register_model(mock_model)

    def test_unregister_runner(
        self, simple_runner, mock_model, manager
    ) -> None:
        """Test unregistering a runner removes its name from the used names."""
        manager.register_runner(simple_runner, mock_model)

        manager.unregister_runner(simple_runner)

        assert simple_runner.name not in manager.named_metadata

    def test_unregister_model(self, mock_model, manager) -> None:
        """Test unregistering a model removes its name from the used names."""
        manager.register_model(mock_model)

        manager.unregister_runner(mock_model)

        assert mock_model.name not in manager.named_metadata

    def test_extract_metadata_recursion_error(self, mocker, manager) -> None:
        """Test extract_metadata handles RecursionError gracefully."""
        mock_obj = mocker.Mock()

        mocker.patch(
            'drytorch.utils.repr_utils.recursive_repr',
            side_effect=RecursionError,
        )
        with pytest.warns(exceptions.RecursionWarning):
            _ = manager.extract_metadata(mock_obj)

    def test_extract_metadata_empty_object(self, manager) -> None:
        """Test extract_metadata on empty object returns class name string."""

        class _Empty:
            pass

        assert manager.extract_metadata(_Empty()) == '_Empty'

    def test_named_metadata_read_only(self, manager) -> None:
        """Test that named_metadata cannot be reassigned."""
        with pytest.raises(AttributeError):
            manager.named_metadata = {}


class TestTracker:
    """Tests for abstract Tracker methods."""

    @pytest.fixture
    def tracker(self) -> Tracker:
        """Set up a test instance."""
        return _SimpleTracker()

    def test_get_current_not_registered(self, tracker) -> None:
        """Error when getting current tracker if none active."""
        with pytest.raises(exceptions.TrackerNotUsedError):
            tracker.get_current()

    def test_get_current(self, tracker, mocker) -> None:
        """Test that get_current returns the active tracker."""
        start_exp = mocker.create_autospec(log_events.StartExperimentEvent)

        tracker.notify(start_exp)

        assert tracker.get_current() is tracker

    def test_get_current_after_stop(self, tracker, mocker) -> None:
        """Test that get_current returns None after stop."""
        start_exp = mocker.create_autospec(log_events.StartExperimentEvent)
        tracker.notify(start_exp)
        stop_exp = mocker.create_autospec(log_events.StopExperimentEvent)

        tracker.notify(stop_exp)

        with pytest.raises(exceptions.TrackerNotUsedError):
            tracker.get_current()

    def test_clean_up_after_stop(self, tracker, mocker) -> None:
        """Test that clean_up removes the tracker from the active trackers."""
        clean_up = mocker.patch.object(tracker, 'clean_up')
        stop_exp = mocker.create_autospec(log_events.StopExperimentEvent)

        tracker.notify(stop_exp)

        clean_up.assert_called_once()

    def test_get_current_after_pause(self, tracker, mocker) -> None:
        """Test that get_current returns None after pause."""
        start_exp = mocker.create_autospec(log_events.StartExperimentEvent)
        tracker.notify(start_exp)
        pause_exp = mocker.create_autospec(log_events.PauseExperimentEvent)

        tracker.notify(pause_exp)

        with pytest.raises(exceptions.TrackerNotUsedError):
            tracker.get_current()

    def test_no_clean_up_after_pause(self, tracker, mocker) -> None:
        """Test that pause does not call clean_up."""
        clean_up = mocker.patch.object(tracker, 'clean_up')
        pause_exp = mocker.create_autospec(log_events.PauseExperimentEvent)

        tracker.notify(pause_exp)

        clean_up.assert_not_called()


class TestEventDispatcher:
    """Test the event dispatcher."""

    @pytest.fixture()
    def tracker(self) -> _SimpleTracker:
        """Set up a test instance."""
        return _SimpleTracker()

    @pytest.fixture()
    def exit_tracker(self) -> Tracker:
        """Set up a test instance."""
        return _ExitTracker()

    @pytest.fixture()
    def dispatcher(self, tracker) -> EventDispatcher:
        """Set up a test instance."""
        dispatcher = EventDispatcher('TestExperiment')
        dispatcher.subscribe(tracker)
        return dispatcher

    def test_register_tracker(self, tracker, dispatcher) -> None:
        """Test that a tracker can be registered to an experiment."""
        assert tracker.__class__.__name__ in dispatcher.named_trackers

    def test_register_duplicate_tracker_raises_error(self, dispatcher) -> None:
        """Test that registering a duplicate tracker raises an error."""
        with pytest.raises(exceptions.TrackerAlreadyRegisteredError):
            dispatcher.subscribe(_SimpleTracker())

    def test_remove_named_tracker(self, tracker, dispatcher) -> None:
        """Test that a registered tracker can be removed by its name."""
        tracker_name = tracker.__class__.__name__

        dispatcher.remove(tracker_name)

        assert tracker_name not in dispatcher.named_trackers

    def test_remove_nonexistent_tracker_raises_error(self, dispatcher) -> None:
        """Test that removing a non-existent tracker raises an error."""
        with pytest.raises(exceptions.TrackerNotUsedError):
            dispatcher.remove('NonexistentTracker')

    def test_publish_event_calls_tracker_notify(
        self, tracker, dispatcher
    ) -> None:
        """Test publishing an event notifies registered trackers."""
        _SimpleEvent.set_auto_publish(dispatcher.publish)

        simple_event = _SimpleEvent()

        assert tracker.last_event is simple_event

    def test_handle_tracker_exceptions(self, dispatcher) -> None:
        """Test handling of tracker exceptions."""
        _UndefinedEvent.set_auto_publish(dispatcher.publish)

        with pytest.warns(exceptions.TrackerExceptionWarning):
            _UndefinedEvent()

    def test_publish_propagates_system_exit(
        self, dispatcher, exit_tracker
    ) -> None:
        """Test KeyboardInterrupt and SystemExit are not swallowed."""
        dispatcher.subscribe(exit_tracker)
        _SimpleEvent.set_auto_publish(dispatcher.publish)

        with pytest.raises(SystemExit):
            _SimpleEvent()

    def test_subscribe_named(self, dispatcher) -> None:
        """Test subscribing with specific names."""
        tracker = _SimpleTracker()

        dispatcher.subscribe(custom_name=tracker)

        assert 'custom_name' in dispatcher.named_trackers
        assert dispatcher.named_trackers['custom_name'] is tracker

    def test_remove_all(self, dispatcher) -> None:
        """Test removing all trackers."""
        dispatcher.remove_all()

        assert not dispatcher.named_trackers

    def test_named_trackers_read_only(self, dispatcher) -> None:
        """Test that named_trackers cannot be reassigned."""
        with pytest.raises(AttributeError):
            dispatcher.named_trackers = {}


def test_extend_default_trackers_iterable() -> None:
    """Test extend_default_trackers accepts an arbitrary Iterable."""
    tracker = _SimpleTracker()

    extend_default_trackers((tracker,))
    try:
        assert DEFAULT_TRACKERS['_SimpleTracker'] is tracker
    finally:
        remove_all_default_trackers()

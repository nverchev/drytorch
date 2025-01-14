"""Tests for the tracking module"""

import pytest

import functools

from src.dry_torch import Experiment
from src.dry_torch import exceptions
from src.dry_torch import log_events
from src.dry_torch.tracking import Tracker


class MockEvent(log_events.Event):
    """Mock Event subclass for testing."""
    pass


class MockTracker(Tracker):
    """Mock Tracker subclass with a defined notify method."""

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        pass


@pytest.fixture
def experiment():
    """Fixture of an experiment."""
    return Experiment(name="TestExperiment")


def test_experiment_initialization(experiment):
    """Test that Experiment initializes with correct properties."""
    assert experiment.name == "TestExperiment"
    assert experiment.dir.exists()
    assert isinstance(experiment.named_trackers, dict)
    assert isinstance(experiment.event_trackers, dict)


def test_register_tracker(experiment):
    """Test that a tracker can be registered to an experiment."""
    tracker = MockTracker()
    experiment.register_tracker(tracker)
    assert "MockTracker" in experiment.named_trackers


def test_register_duplicate_tracker_raises_error(experiment):
    """Test that registering a duplicate tracker raises an error."""
    tracker = MockTracker()
    experiment.register_tracker(tracker)
    with pytest.raises(exceptions.TrackerAlreadyRegisteredError):
        experiment.register_tracker(tracker)


def test_remove_named_tracker(experiment):
    """Test that a registered tracker can be removed by name."""
    tracker = MockTracker()
    experiment.register_tracker(tracker)
    experiment.remove_named_tracker("MockTracker")
    assert "MockTracker" not in experiment.named_trackers


def test_remove_nonexistent_tracker_raises_error(experiment):
    """Test that removing a non-existent tracker raises an error."""
    with pytest.raises(exceptions.TrackerNotRegisteredError):
        experiment.remove_named_tracker("NonexistentTracker")


def test_publish_event_calls_tracker_notify(mocker, experiment):
    """Test publishing an event calls notify on registered trackers."""
    event = MockEvent()
    tracker = mocker.create_autospec(MockTracker, instance=True)
    tracker.defined_events.return_value = [event.__class__]
    # create_autospec has problems with functools singledispatchmethod
    tracker.notify = mocker.Mock()
    experiment.register_tracker(tracker)
    experiment.publish(event)
    tracker.notify.assert_called_once_with(event)


def test_start_and_stop_experiment(mocker):
    """Test starting and stopping an experiment."""
    mock_event_start = mocker.patch.object(log_events, "StartExperiment")
    mock_event_stop = mocker.patch.object(log_events, "StopExperiment")

    exp = Experiment(name="TestExp")
    exp.start()
    mock_event_start.assert_called_once_with("TestExp")

    exp.stop()
    mock_event_stop.assert_called_once_with("TestExp")


def test_get_config_no_config_error():
    """Test NoConfigError is raised if config is accessed when unavailable."""
    Experiment._current_config = None  # Ensure no config is set
    with pytest.raises(exceptions.NoConfigError):
        Experiment.get_config()

"""Tests for the tracking module."""
from typing import Optional

import pytest

import functools

from src.dry_torch import Experiment
from src.dry_torch import exceptions
from src.dry_torch import log_events
from src.dry_torch.tracking import Tracker


class _SimpleEvent(log_events.Event):
    """Simple Event subclass for testing."""
    pass


class _SimpleTracker(Tracker):
    """Simple Tracker subclass with a defined notify method."""
    last_event: Optional[log_events.Event] = None

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        self.last_event = event


class TestExperiment:
    """Test the Experiment class."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path) -> None:
        """Set up an experiment."""
        self.name = 'TestExperiment'
        self.par_dir = tmp_path
        self.experiment = Experiment[None](self.name, self.par_dir)
        self.tracker = _SimpleTracker()
        self.experiment.register_tracker(self.tracker)
        return

    def test_experiment_initialization(self):
        """Test that Experiment initializes with correct properties."""
        assert self.experiment.name == self.name
        assert self.experiment.dir.exists()

    def test_register_tracker(self):
        """Test that a tracker can be registered to an experiment."""
        assert self.tracker.__class__.__name__ in self.experiment.named_trackers

    def test_register_duplicate_tracker_raises_error(self):
        """Test that registering a duplicate tracker raises an error."""
        with pytest.raises(exceptions.TrackerAlreadyRegisteredError):
            self.experiment.register_tracker(_SimpleTracker())

    def test_remove_named_tracker(self):
        """Test that a registered tracker can be removed by name."""
        tracker_name = self.tracker.__class__.__name__
        self.experiment.remove_named_tracker(tracker_name)
        assert tracker_name not in self.experiment.named_trackers

    def test_remove_nonexistent_tracker_raises_error(self):
        """Test that removing a non-existent tracker raises an error."""
        with pytest.raises(exceptions.TrackerNotRegisteredError):
            self.experiment.remove_named_tracker('NonexistentTracker')

    def test_start_and_stop_experiment(self, mocker):
        """Test starting and stopping an experiment."""
        mock_event_start = mocker.patch.object(log_events, 'StartExperiment')
        self.experiment.start()
        mock_event_start.assert_called_once_with(self.name,
                                                 self.par_dir / self.name)

        mock_event_stop = mocker.patch.object(log_events, 'StopExperiment')
        self.experiment.stop()
        mock_event_stop.assert_called_once_with(self.name)

    def test_publish_event_calls_tracker_notify(self):
        """Test publishing an event calls notify on registered trackers."""
        simple_event = _SimpleEvent()
        self.experiment.publish(event=simple_event)
        assert self.tracker.last_event is simple_event


def test_get_config_no_config_error():
    """Test NoConfigError is raised if config is accessed when unavailable."""
    Experiment._current_config = None  # Ensure no config is set
    with pytest.raises(exceptions.NoConfigError):
        Experiment.get_config()

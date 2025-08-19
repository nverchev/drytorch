"""Tests for the "visdom" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('visdom'):
    pytest.skip('visdom not available', allow_module_level=True)

from collections.abc import Generator

import numpy as np

from drytorch.core import exceptions
from drytorch.trackers.visdom import VisdomOpts, VisdomPlotter


class TestVisdomPlotter:
    """Tests for the VisdomPlotter tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up a test environment."""
        self.viz_instance = mocker.Mock()
        self.visdom_mock = mocker.patch('visdom.Visdom')
        self.visdom_mock.return_value = self.viz_instance
        return

    @pytest.fixture
    def tracker(self) -> VisdomPlotter:
        """Set up the instance."""
        return VisdomPlotter(opts=VisdomOpts(title='test title'))

    @pytest.fixture
    def tracker_started(
        self,
        tracker,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[VisdomPlotter, None, None]:
        """Set up started instance."""
        tracker.notify(start_experiment_mock_event)
        yield tracker

        tracker.notify(stop_experiment_mock_event)
        return

    def test_init(self, tracker) -> None:
        """Test initialization."""
        assert isinstance(tracker.server, str)
        assert isinstance(tracker.port, int)
        assert isinstance(tracker.opts, dict)  # cannot specify TypedDict

    def test_viz_property_fails(self, tracker) -> None:
        """Test viz property raises ac exception when accessed outside scope."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _ = tracker.viz

    def test_viz_property_succeeds(self, tracker_started) -> None:
        """Test viz property returns a visdom instance when initialized."""
        assert tracker_started.viz is self.viz_instance

    def test_clean_up(self, tracker_started) -> None:
        """Test cleanup sets the viz attribute to None."""
        tracker_started.clean_up()
        assert tracker_started._viz is None

    def test_notify_start_experiment(
        self, tracker, start_experiment_mock_event, example_run_id
    ) -> None:
        """Test StartExperiment notification."""
        tracker.notify(start_experiment_mock_event)
        self.visdom_mock.assert_called_once()
        self.viz_instance.close.assert_called_once_with(
            env=f'{start_experiment_mock_event.exp_name}_{example_run_id}'
        )

    def test_notify_start_experiment_fails(
        self, tracker, start_experiment_mock_event
    ) -> None:
        """Test StartExperiment notification with a connection error."""
        self.visdom_mock.side_effect = ConnectionError('Connection failed')
        with pytest.raises(exceptions.TrackerError):
            tracker.notify(start_experiment_mock_event)

    def test_notify_stop_experiment(
        self, tracker_started, stop_experiment_mock_event
    ) -> None:
        """Test StopExperiment notification."""
        assert tracker_started._viz is not None

        tracker_started.notify(stop_experiment_mock_event)
        assert tracker_started._viz is None

    def test_plot_metric_single_point(
        self,
        tracker_started,
        example_source_name,
        example_model_name,
        example_loss_name,
    ) -> None:
        """Test plotting a single data point (scatter plot)."""
        sourced_array = {example_source_name: np.array([[1, 0.85]])}
        win = tracker_started._plot_metric(
            example_model_name, example_loss_name, **sourced_array
        )
        self.viz_instance.scatter.assert_any_call(
            None, win=win, update='remove', name=example_source_name
        )
        assert self.viz_instance.scatter.call_count == 2

    def test_plot_metric_multiple_points(
        self,
        tracker_started,
        example_source_name,
        example_model_name,
        example_loss_name,
    ) -> None:
        """Test plotting multiple data points (line plot)."""
        sourced_array = {example_source_name: np.array([[1, 0.85], [2, 2.2]])}
        win = tracker_started._plot_metric(
            example_model_name, example_loss_name, **sourced_array
        )
        self.viz_instance.scatter.assert_called_once_with(
            None, win=win, update='remove', name=example_source_name
        )
        assert self.viz_instance.line.call_count == 1

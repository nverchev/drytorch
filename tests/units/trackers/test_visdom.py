"""Tests for the "visdom" module."""

import pytest
try:
    import visdom
except ImportError:
    pytest.skip('visdom not available', allow_module_level=True)

from typing import Generator

import numpy as np

from drytorch import exceptions
from drytorch.trackers.visdom import VisdomPlotter, VisdomOpts


class TestVisdomPlotter:
    """Tests for the VisdomPlotter tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up test environment."""
        self.viz_instance = mocker.Mock()
        self.visdom_mock = mocker.patch('visdom.Visdom')
        self.visdom_mock.return_value = self.viz_instance
        return

    @pytest.fixture
    def tracker(self) -> VisdomPlotter:
        """Set up instance."""
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
        """Test viz property raises exception when accessed outside scope."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _ = tracker.viz

    def test_viz_property_succeeds(self, tracker_started) -> None:
        """Test viz property returns visdom instance when initialized."""
        assert tracker_started.viz is self.viz_instance

    def test_clean_up(self, tracker_started) -> None:
        """Test cleanup sets viz to None."""
        tracker_started.clean_up()
        assert tracker_started._viz is None

    def test_notify_start_experiment(self,
                                     tracker,
                                     start_experiment_mock_event) -> None:
        """Test StartExperiment notification."""
        tracker.notify(start_experiment_mock_event)
        self.visdom_mock.assert_called_once()
        self.viz_instance.close.assert_called_once_with(
            env=start_experiment_mock_event.exp_name
        )

    def test_notify_start_experiment_fails(self,
                                           tracker,
                                           start_experiment_mock_event) -> None:
        """Test StartExperiment notification with connection error."""
        self.visdom_mock.side_effect = ConnectionError("Connection failed")
        with pytest.raises(exceptions.TrackerException):
            tracker.notify(start_experiment_mock_event)

    def test_notify_stop_experiment(self,
                                    tracker_started,
                                    stop_experiment_mock_event) -> None:
        """Test StopExperiment notification."""
        assert tracker_started._viz is not None

        tracker_started.notify(stop_experiment_mock_event)
        assert tracker_started._viz is None

    def test_plot_metric_single_point(self,
                                      tracker_started,
                                      example_source_name,
                                      example_model_name,
                                      example_loss_name) -> None:
        """Test plotting a single data point (scatter plot)."""
        sourced_array = {example_source_name: np.array([[1, 0.85]])}
        win = tracker_started._plot_metric(example_model_name,
                                           example_loss_name,
                                           **sourced_array)
        self.viz_instance.scatter.assert_any_call(
            None, win=win, update='remove', name=example_source_name
        )
        assert self.viz_instance.scatter.call_count == 2

    def test_plot_metric_multiple_points(self,
                                         tracker_started,
                                         example_source_name,
                                         example_model_name,
                                         example_loss_name) -> None:
        """Test plotting a multiple data points (line plot)."""
        sourced_array = {example_source_name: np.array([[1, 0.85],
                                                        [2, 2.2]])}
        win = tracker_started._plot_metric(example_model_name,
                                           example_loss_name,
                                           **sourced_array)
        self.viz_instance.scatter.assert_called_once_with(
            None, win=win, update='remove', name=example_source_name
        )
        assert self.viz_instance.line.call_count == 1

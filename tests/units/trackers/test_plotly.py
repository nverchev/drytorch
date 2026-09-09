"""Tests for the "plotly" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('plotly'):
    pytest.skip('plotly not available', allow_module_level=True)

import numpy as np
import plotly.graph_objs as go

from drytorch.core import log_events
from drytorch.trackers.plotly import PlotlyPlotter


class TestPlotlyPlotter:
    """Tests for the PlotlyPlotter class."""

    @pytest.fixture
    def tracker(self) -> PlotlyPlotter:
        """Set up the instance."""
        return PlotlyPlotter()

    def test_plot_single_point(
        self,
        tracker: PlotlyPlotter,
        example_loss_name: str,
        example_source_name: str,
        example_model_name: str,
        mocker,
    ) -> None:
        """Test plot_metric with a single datapoint."""
        # Prepare
        mock_show = mocker.patch.object(go.Figure, 'show')
        model_name = example_model_name
        data = np.array([[1, 0.8]])
        sourced_array = {example_source_name: data}

        # Trigger
        fig = tracker._plot_metric(
            model_name, example_loss_name, **sourced_array
        )

        # Assert
        assert fig.data[0].marker.symbol == 'diamond'
        assert fig.data[0].marker.size == 20
        mock_show.assert_not_called()

    def test_plot_multiple_points(
        self,
        tracker: PlotlyPlotter,
        example_loss_name: str,
        example_source_name: str,
        example_model_name: str,
        mocker,
    ) -> None:
        """Test plot_metric with multiple points."""
        # Prepare
        mock_show = mocker.patch.object(go.Figure, 'show')
        model_name = example_model_name
        multi_points = np.array([[1, 0.7], [2, 0.8], [3, 0.85]])
        sourced_array = {example_source_name: multi_points}

        # Trigger
        fig = tracker._plot_metric(
            model_name, example_loss_name, **sourced_array
        )

        # Assert
        assert fig.data[0].marker.symbol is None
        assert len(fig.data[0].x) == 3
        mock_show.assert_not_called()

    def test_plot_renders_figures(
        self,
        tracker: PlotlyPlotter,
        example_model_name: str,
        example_source_name: str,
        mocker,
    ) -> None:
        """Verify that plot() renders each returned figure."""
        # Prepare
        mock_show = mocker.patch.object(go.Figure, 'show')
        metric_event = log_events.MetricEvent(
            model_name=example_model_name,
            source_name=example_source_name,
            epoch=1,
            metrics={'loss': 0.5, 'accuracy': 0.8},
        )
        end_epoch_event = log_events.EndEpochEvent(
            model_name=example_model_name,
            source_name=example_source_name,
            epoch=1,
        )

        # Trigger
        tracker.notify(metric_event)
        tracker.notify(end_epoch_event)
        figures = tracker.plot(example_model_name)

        # Assert
        assert len(figures) == 2
        assert mock_show.call_count == len(figures)

    def test_training_neither_renders_nor_builds(
        self,
        tracker: PlotlyPlotter,
        example_model_name: str,
        example_source_name: str,
        mocker,
    ) -> None:
        """Verify training events neither render nor build figures."""
        # Prepare
        mock_show = mocker.patch.object(go.Figure, 'show')
        mock_init = mocker.patch.object(
            go.Figure, '__init__', return_value=None
        )
        metric_event = log_events.MetricEvent(
            model_name=example_model_name,
            source_name=example_source_name,
            epoch=1,
            metrics={'loss': 0.5, 'accuracy': 0.8},
        )
        end_epoch_event = log_events.EndEpochEvent(
            model_name=example_model_name,
            source_name=example_source_name,
            epoch=1,
        )

        # Trigger
        tracker.notify(metric_event)
        tracker.notify(end_epoch_event)

        # Assert
        mock_show.assert_not_called()
        mock_init.assert_not_called()

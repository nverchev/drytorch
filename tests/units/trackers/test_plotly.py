"""Tests for the "plotly" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('plotly'):
    pytest.skip('plotly not available', allow_module_level=True)

import numpy as np

from drytorch.trackers.plotly import PlotlyPlotter


class TestPlotlyPlotter:
    """Tests for the PlotlyPlotter class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test."""
        self.go_mock = mocker.patch('drytorch.trackers.plotly.go')
        mock_figure = mocker.Mock()
        mock_marker = mocker.Mock()
        self.go_mock.Figure.return_value = mock_figure
        self.go_mock.Marker.return_value = mock_marker
        self.mock_figure = mock_figure

    @pytest.fixture
    def tracker(self) -> PlotlyPlotter:
        """Set up the instance."""
        return PlotlyPlotter()

    def test_plot_single_point(
        self,
        tracker,
        example_loss_name,
        example_source_name,
        example_model_name,
    ) -> None:
        """Test plot_metric with a single datapoint."""
        model_name = example_model_name
        data = np.array([[1, 0.8]])
        sourced_array = {example_source_name: data}
        tracker._plot_metric(model_name, example_loss_name, **sourced_array)
        self.go_mock.scatter.Marker.assert_called_once_with(symbol=24, size=20)
        self.go_mock.Figure.assert_called_once()

    def test_plot_multiple_points(
        self,
        tracker,
        example_loss_name,
        example_source_name,
        example_model_name,
    ) -> None:
        """Test plot_metric with multiple points."""
        model_name = example_model_name
        multi_points = np.array([[1, 0.7], [2, 0.8], [3, 0.85]])
        sourced_array = {example_source_name: multi_points}
        tracker._plot_metric(model_name, example_model_name, **sourced_array)
        self.go_mock.scatter.Marker.assert_not_called()
        self.go_mock.Figure.assert_called_once()

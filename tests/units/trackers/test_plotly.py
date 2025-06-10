"""Tests for the PlotlyPlotter class."""

import pytest

import numpy as np

from dry_torch.trackers.plotly import PlotlyPlotter


class TestPlotlyPlotter:
    """Tests for the PlotlyPlotter class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test."""
        self.go_mock = mocker.patch('dry_torch.trackers.plotly.go')
        mock_figure = mocker.Mock()
        self.go_mock.Figure.return_value = mock_figure
        self.mock_figure = mock_figure

    @pytest.fixture
    def tracker(self) -> PlotlyPlotter:
        """Set up the instance."""
        return PlotlyPlotter()

    def test_plot_metric(self, tracker) -> None:
        """Test plotting metric with single and multiple points."""
        model_name = 'test_model'
        metric_name = 'accuracy'

        # Test data: single point and multiple points
        single_point = np.array([[1, 0.8]])
        multi_points = np.array([[1, 0.7], [2, 0.8], [3, 0.85]])
        sourced_array = {'val': single_point, 'train': multi_points}

        result = tracker._plot_metric(model_name, metric_name, **sourced_array)

        # Verify marker created for single point
        self.go_mock.scatter.Marker.assert_called_once_with(symbol=24, size=20)

        # Verify two scatter plots created
        assert self.go_mock.Scatter.call_count == 2

        # Verify layout and figure creation
        self.go_mock.Layout.assert_called_once_with(
            title=model_name,
            xaxis=dict(title='Epoch'),
            yaxis=dict(title=metric_name)
        )
        self.go_mock.Figure.assert_called_once()
        self.mock_figure.show.assert_called_once()

        assert result == self.mock_figure

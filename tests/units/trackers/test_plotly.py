"""Tests for the Plotly plotter in the trackers package."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from dry_torch import log_events
from dry_torch.trackers.plotly import PlotlyPlotter


@pytest.fixture
def mock_plotly_figure():
    """Fixture that mocks plotly.graph_objs.Figure."""
    with patch('plotly.graph_objs.Figure', autospec=True) as mock:
        # Create a mock Figure instance
        figure_instance = MagicMock()
        figure_instance.show = MagicMock()
        mock.return_value = figure_instance
        yield mock


@pytest.fixture
def mock_plotly_scatter():
    """Fixture that mocks plotly.graph_objs.Scatter."""
    with patch('plotly.graph_objs.Scatter', autospec=True) as mock:
        yield mock


@pytest.fixture
def mock_plotly_bar():
    """Fixture that mocks plotly.graph_objs.Bar."""
    with patch('plotly.graph_objs.Bar', autospec=True) as mock:
        yield mock


@pytest.fixture
def mock_plotly_layout():
    """Fixture that mocks plotly.graph_objs.Layout."""
    with patch('plotly.graph_objs.Layout', autospec=True) as mock:
        yield mock


@pytest.fixture
def mock_plotly_marker():
    """Fixture that mocks plotly.graph_objs.scatter.Marker."""
    with patch('plotly.graph_objs.scatter.Marker', autospec=True) as mock:
        yield mock


class TestPlotlyPlotter:
    """Tests for the PlotlyPlotter class."""

    def test_initialization(self):
        """Test basic initialization."""
        plotter = PlotlyPlotter()
        assert plotter._model_names == ()
        assert plotter._metric_names == ()
        assert plotter._start == 1

        # Test with parameters
        plotter = PlotlyPlotter(
            model_names=["model1", "model2"],
            metric_names=["loss", "accuracy"],
            start=5
        )
        assert plotter._model_names == ["model1", "model2"]
        assert plotter._metric_names == ["loss", "accuracy"]
        assert plotter._start == 5

    def test_plot_metric_multipoint(
            self,
            mock_plotly_figure,
            mock_plotly_scatter,
            mock_plotly_layout
    ):
        """Test _plot_metric method with multi-point data."""
        # Initialize plotter
        plotter = PlotlyPlotter()

        # Test with multi-point data
        data = np.array([[1, 0.5], [2, 0.4], [3, 0.3]])

        result = plotter._plot_metric("test_model", "loss",
                                      **{"test_source": data})

        # Verify Scatter was created with correct params
        mock_plotly_scatter.assert_called_once_with(
            x=data[:, 0], y=data[:, 1], name="test_source"
        )

        # Verify Layout was created with correct params
        mock_plotly_layout.assert_called_once_with(
            title="test_model",
            xaxis=dict(title='Epoch'),
            yaxis=dict(title='loss')
        )

        # Verify Figure was created and shown
        mock_plotly_figure.assert_called_once()
        result.show.assert_called_once()

    def test_plot_metric_single_point(
            self,
            mock_plotly_figure,
            mock_plotly_scatter,
            mock_plotly_layout,
            mock_plotly_marker
    ):
        """Test _plot_metric method with single point data."""
        # Initialize plotter
        plotter = PlotlyPlotter()

        # Test with single-point data
        data = np.array([[1, 0.7]])

        plotter._plot_metric("test_model", "loss", **{"single_point": data})

        # Verify Marker was created with correct params
        mock_plotly_marker.assert_called_once_with(symbol=24, size=20)

        # Verify Scatter was created with marker mode
        mock_plotly_scatter.assert_called_once_with(
            x=data[:, 0],
            y=data[:, 1],
            mode='markers',
            marker=mock_plotly_marker.return_value,
            name="single_point"
        )

    def test_plot_metric_multiple_sources(
            self,
            mock_plotly_figure,
            mock_plotly_scatter,
            mock_plotly_layout,
            mock_plotly_marker
    ):
        """Test _plot_metric method with multiple sources."""
        # Initialize plotter
        plotter = PlotlyPlotter()

        # Test with multiple sources
        data1 = np.array([[1, 0.5], [2, 0.4], [3, 0.3]])
        data2 = np.array([[1, 0.7]])

        plotter._plot_metric(
            "test_model", "loss",
            source1=data1, source2=data2
        )

        # Verify Scatter was called twice (once for each source)
        assert mock_plotly_scatter.call_count == 2

        # Verify the correct data was used for each call
        # First call for multi-point
        # Second call for single point with marker
        calls = mock_plotly_scatter.call_args_list

        # Check first call (regular line)
        assert np.array_equal(calls[0][1]['x'], data1[:, 0])
        assert np.array_equal(calls[0][1]['y'], data1[:, 1])
        assert calls[0][1]['name'] == 'source1'

        # Check second call (scatter)
        assert np.array_equal(calls[1][1]['x'], data2[:, 0])
        assert np.array_equal(calls[1][1]['y'], data2[:, 1])
        assert calls[1][1]['name'] == 'source2'
        assert calls[1][1]['mode'] == 'markers'

    @patch('plotly.graph_objs.Figure', autospec=True)
    def test_end_epoch_notification(self, mock_figure):
        """Test notification with EndEpoch event."""
        # Set up mocks
        figure_instance = MagicMock()
        figure_instance.show = MagicMock()
        mock_figure.return_value = figure_instance

        # Initialize plotter with metrics
        plotter = PlotlyPlotter()
        plotter.model_metrics = {
            "test_model": {
                "test_source": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3]})
            }
        }

        # Create and notify with end epoch event
        event = log_events.EndEpoch(
            source_name="test_source",
            model_name="test_model",
            epoch=3
        )

        # Should create figure and plot
        plotter.notify(event)

        # Verify Figure was created
        mock_figure.assert_called_once()
        figure_instance.show.assert_called_once()

    @patch('plotly.graph_objs.Figure', autospec=True)
    def test_plot_method(self, mock_figure):
        """Test the plot method."""
        # Set up mocks
        figure_instance = MagicMock()
        figure_instance.show = MagicMock()
        mock_figure.return_value = figure_instance

        # Create a plotter with mock data
        plotter = PlotlyPlotter()
        plotter.model_metrics = {
            "test_model": {
                "source1": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3],
                                        "accuracy": [0.7, 0.8, 0.9]}),
                "source2": ([1, 2], {"loss": [0.6, 0.5]})
            }
        }

        # Test plot method
        result = plotter.plot("test_model")

        # Should create figures for both metrics
        assert mock_figure.call_count == 2
        assert figure_instance.show.call_count == 2

        # Result should be a list of figures
        assert len(result) == 2
        assert result[0] == figure_instance
        assert result[1] == figure_instance

        # Reset mocks
        mock_figure.reset_mock()
        figure_instance.show.reset_mock()

        # Test with filtered metrics
        result = plotter.plot("test_model", metric_names=["loss"])

        # Should only create one figure
        assert mock_figure.call_count == 1
        assert figure_instance.show.call_count == 1
        assert len(result) == 1

        # Reset mocks
        mock_figure.reset_mock()
        figure_instance.show.reset_mock()

        # Test with filtered sources
        result = plotter.plot("test_model", source_names=["source1"])

        # Should still create two figures (one for each metric)
        assert mock_figure.call_count == 2
        assert figure_instance.show.call_count == 2
        assert len(result) == 2

        # Test with nonexistent model
        with pytest.raises(Exception):
            plotter.plot("nonexistent_model")

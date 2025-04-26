"""Tests for the matplotlib plotter in the trackers package."""

import math
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, call

from dry_torch import log_events
from dry_torch.trackers.base_classes import MetricLoader
from dry_torch.trackers.matplotlib import MatPlotter


class MockMetricLoader(MetricLoader):
    """Mock implementation of MetricLoader for testing."""

    def __init__(self):
        super().__init__()
        self.metrics_data = {
            "test_model": {
                "test_source": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3],
                                            "accuracy": [0.7, 0.8, 0.9]})
            }
        }

    def _load_metrics(self, model_name, max_epoch=-1):
        return self.metrics_data.get(model_name, {})


@pytest.fixture
def mock_plt():
    """Fixture that mocks matplotlib.pyplot."""
    with patch('matplotlib.pyplot', autospec=True) as mock:
        yield mock


@pytest.fixture
def mock_figure():
    """Fixture that creates a mock Figure with necessary methods."""
    mock = MagicMock()
    mock.canvas = MagicMock()
    mock.canvas.draw = MagicMock()
    mock.canvas.flush_events = MagicMock()
    mock.add_subplot.return_value = MagicMock()
    mock.tight_layout = MagicMock()
    mock.suptitle = MagicMock()
    return mock


class TestMatPlotter:
    """Tests for the MatPlotter class."""

    def test_initialization(self, mock_plt):
        """Test basic initialization."""
        plotter = MatPlotter()
        assert plotter.model_figures == {}
        assert plotter._model_names == ()
        assert plotter._metric_names == ()
        assert plotter._start == 1

        # Check that plt.ion() was called
        mock_plt.ion.assert_called_once()

        # Test with parameters
        loader = MockMetricLoader()
        plotter = MatPlotter(
            model_names=["model1", "model2"],
            metric_names=["loss", "accuracy"],
            metric_loader=loader,
            start=5
        )
        assert plotter._model_names == ["model1", "model2"]
        assert plotter._metric_names == ["loss", "accuracy"]
        assert plotter._start == 5
        assert plotter.metric_loader is loader

    @patch('matplotlib.pyplot.Figure', autospec=True)
    @patch('matplotlib.pyplot.show', autospec=True)
    def test_prepare_layout(self, mock_show, mock_figure_class, mock_figure,
                            mock_plt):
        """Test _prepare_layout method."""
        # Set up the mock Figure to be returned by plt.Figure()
        mock_figure_class.return_value = mock_figure

        # Create subplot axes mocks
        mock_axes = {}
        for metric in ["loss", "accuracy"]:
            mock_ax = MagicMock()
            mock_axes[metric] = mock_ax

        # Configure mock_figure.add_subplot to return the appropriate mock axis
        def add_subplot_side_effect(n_rows, n_cols, pos):
            if pos == (0, 0):
                return mock_axes["loss"]
            else:
                return mock_axes["accuracy"]

        mock_figure.add_subplot.side_effect = add_subplot_side_effect

        # Initialize plotter and test _prepare_layout
        plotter = MatPlotter()
        plotter._prepare_layout("test_model", ["loss", "accuracy"])

        # Verify expected calls
        mock_figure_class.assert_called_once()
        mock_figure.suptitle.assert_called_once_with("test_model", fontsize=16)
        mock_figure.tight_layout.assert_called_once()

        # Check that the model was added to model_figures
        assert "test_model" in plotter.model_figures
        assert plotter.model_figures["test_model"][0] == mock_figure
        assert set(plotter.model_figures["test_model"][1].keys()) == {"loss",
                                                                      "accuracy"}

        # Check that show was called
        mock_show.assert_called_once_with(block=False)

        # Test with more metrics to check grid calculation
        mock_figure.reset_mock()
        mock_show.reset_mock()

        plotter = MatPlotter()
        plotter._prepare_layout("test_model",
                                ["loss", "accuracy", "precision", "recall",
                                 "f1"])

        # Verify layout calculations (should be 3x2 grid for 5 metrics)
        mock_figure.add_subplot.assert_any_call(3, 2, (0, 0))  # First call
        mock_figure.add_subplot.assert_any_call(3, 2, (2, 1))  # Last call

        # Check layout with single metric
        mock_figure.reset_mock()
        plotter = MatPlotter()
        plotter._prepare_layout("test_model", ["loss"])

        # Should be a 1x1 grid
        mock_figure.add_subplot.assert_called_once_with(1, 1, (0, 0))

    @patch('matplotlib.pyplot.Figure', autospec=True)
    def test_plot_metric(self, mock_figure_class, mock_figure, mock_plt):
        """Test _plot_metric method."""
        # Set up mocks
        mock_figure_class.return_value = mock_figure
        mock_ax = MagicMock()
        mock_line = MagicMock()
        mock_line.get_label.return_value = "test_source"
        mock_ax.get_lines.return_value = [mock_line]

        # Initialize plotter and setup model_figures
        plotter = MatPlotter()
        plotter.model_figures = {
            "test_model": (mock_figure, {"loss": mock_ax})
        }

        # Test with existing line
        data = np.array([[1, 0.5], [2, 0.4], [3, 0.3]])
        result = plotter._plot_metric("test_model", "loss",
                                      **{"test_source": data})

        # Verify line was updated
        mock_line.set_xdata.assert_called_once_with(data[:, 0])
        mock_line.set_ydata.assert_called_once_with(data[:, 1])

        # Verify axis methods were called
        mock_ax.relim.assert_called_once()
        mock_ax.autoscale_view.assert_called_once()
        mock_ax.legend.assert_called_once()

        # Verify figure was drawn
        mock_figure.canvas.draw.assert_called_once()
        mock_figure.canvas.flush_events.assert_called_once()

        # Check return value
        assert result == (mock_figure, mock_ax)

        # Reset mocks
        mock_ax.reset_mock()
        mock_line.reset_mock()
        mock_figure.reset_mock()

        # Test with new source (multiple points)
        mock_ax.get_lines.return_value = []  # No existing lines
        data = np.array([[1, 0.6], [2, 0.5]])

        plotter._plot_metric("test_model", "loss", **{"new_source": data})

        # Verify new line was created
        mock_ax.plot.assert_called_once_with(data[:, 0], data[:, 1],
                                             label="new_source")

        # Reset mocks
        mock_ax.reset_mock()

        # Test with single point (should create scatter)
        data = np.array([[1, 0.7]])

        plotter._plot_metric("test_model", "loss", **{"single_point": data})

        # Verify scatter was created
        mock_ax.scatter.assert_called_once_with(
            data[:, 0], data[:, 1], s=200, label="single_point", marker='D'
        )

        # Test with collections to remove
        mock_ax.reset_mock()
        mock_collection = MagicMock()
        mock_ax.collections = [mock_collection]

        plotter._plot_metric("test_model", "loss",
                             **{"test_source": np.array([[1, 0.5]])})

        # Verify collection was removed
        mock_collection.remove.assert_called_once()

    def test_close_all(self, mock_plt):
        """Test close_all method."""
        # Create mock figures
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()

        # Initialize plotter and setup model_figures
        plotter = MatPlotter()
        plotter.model_figures = {
            "model1": (mock_fig1, {"loss": MagicMock()}),
            "model2": (mock_fig2, {"accuracy": MagicMock()})
        }

        # Call close_all
        plotter.close_all()

        # Verify plt.close was called for each figure
        mock_plt.close.assert_has_calls([call(mock_fig1), call(mock_fig2)],
                                        any_order=True)

        # Verify model_figures was cleared
        assert plotter.model_figures == {}

    @patch('matplotlib.pyplot.Figure', autospec=True)
    @patch('matplotlib.pyplot.show', autospec=True)
    def test_end_epoch_notification(self, mock_show, mock_figure_class,
                                    mock_figure, mock_plt):
        """Test notification with EndEpoch event."""
        # Set up mocks
        mock_figure_class.return_value = mock_figure
        mock_ax = MagicMock()
        mock_figure.add_subplot.return_value = mock_ax

        # Initialize plotter with metrics
        plotter = MatPlotter()
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

        # Verify figure was created
        assert "test_model" in plotter.model_figures

        # Verify show was called
        mock_show.assert_called_once_with(block=False)

    def test_plot_method(self, mock_plt, mock_figure):
        """Test the plot method."""
        with patch('matplotlib.pyplot.Figure', return_value=mock_figure):
            # Create a plotter with mock data
            plotter = MatPlotter()
            plotter.model_metrics = {
                "test_model": {
                    "source1": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3],
                                            "accuracy": [0.7, 0.8, 0.9]}),
                    "source2": ([1, 2], {"loss": [0.6, 0.5]})
                }
            }

            # Test plot method
            result = plotter.plot("test_model")

            # Should have created figures
            assert "test_model" in plotter.model_figures

            # Should return two plots (one for each metric)
            assert len(result) == 2

            # Each result should be a tuple of (figure, axes)
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2

            # Test with filtered metrics
            mock_plt.reset_mock()
            plotter.model_figures.clear()

            result = plotter.plot("test_model", metric_names=["loss"])

            # Should only have one plot
            assert len(result) == 1

            # Test with filtered sources
            mock_plt.reset_mock()
            plotter.model_figures.clear()

            result = plotter.plot("test_model", source_names=["source1"])

            # Should still have two plots (one for each metric)
            assert len(result) == 2

            # But each plot should only have one source
            # (This is harder to test without inspecting internal details)

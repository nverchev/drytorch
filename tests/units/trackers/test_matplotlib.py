"""Tests for the "matplotlib" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('matplotlib'):
    pytest.skip('matplotlib not available', allow_module_level=True)

import numpy as np

from drytorch.trackers.matplotlib import MatPlotter


class TestMatPlotter:
    """Tests for the MatPlotter class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test."""
        self.plt_mock = mocker.patch(
            'drytorch.trackers.matplotlib.plt', autospec=True
        )
        self.figure_mock = mocker.patch(
            'drytorch.trackers.matplotlib.figure', autospec=True
        )

        # create a mock figure with all necessary attributes
        mock_fig = mocker.Mock()
        mock_fig.canvas = mocker.Mock()
        mock_fig.canvas.draw = mocker.Mock()
        mock_fig.canvas.flush_events = mocker.Mock()
        mock_fig.add_subplot.return_value = mocker.Mock()
        mock_fig.tight_layout = mocker.Mock()
        mock_fig.suptitle = mocker.Mock()

        # mock axes with necessary attributes
        mock_ax = mocker.Mock()
        mock_ax.collections = []
        mock_ax.get_lines.return_value = []
        mock_ax.scatter = mocker.Mock()
        mock_ax.plot = mocker.Mock()
        mock_ax.relim = mocker.Mock()
        mock_ax.autoscale_view = mocker.Mock()
        mock_ax.legend = mocker.Mock()

        mock_fig.add_subplot.return_value = mock_ax
        self.figure_mock.Figure.return_value = mock_fig
        self.plt_mock.show = mocker.Mock()
        self.plt_mock.close = mocker.Mock()
        self.mock_fig = mock_fig
        self.mock_ax = mock_ax

    @pytest.fixture
    def data(self) -> np.ndarray:
        """Create test data."""
        return np.array([[1, 0.8], [2, 0.85], [3, 0.9]])

    @pytest.fixture
    def sourced_array(self, data, example_source_name) -> dict[str, np.ndarray]:
        """Return sourced data."""
        return {example_source_name: data}

    @pytest.fixture
    def tracker(self) -> MatPlotter:
        """Set up the instance."""
        return MatPlotter()

    @pytest.fixture
    def tracker_with_layout(
        self, tracker, example_loss_name, example_model_name
    ) -> MatPlotter:
        """Set up the instance with the layout."""
        tracker._prepare_layout(example_model_name, [example_loss_name])
        return tracker

    def test_initialization(self, tracker) -> None:
        """Test initialization."""
        self.plt_mock.ion.assert_called_once()
        assert tracker._model_figure == {}

    def test_prepare_layout(self, tracker, example_model_name) -> None:
        """Test layout preparation with multiple metrics."""
        model_name = example_model_name
        metric_names = ['accuracy', 'loss', 'f1_score']
        tracker._prepare_layout(model_name, metric_names)

        # should create 2x2 grid (math.ceil(sqrt(3)) = 2)
        expected_calls = [
            ((2, 2, 1),),  # first subplot
            ((2, 2, 2),),  # second subplot
            ((2, 2, 3),),  # third subplot
        ]
        assert self.mock_fig.add_subplot.call_count == 3

        for i, call_args in enumerate(self.mock_fig.add_subplot.call_args_list):
            assert call_args == expected_calls[i]

    def test_prepare_layout_already_exists(
        self, tracker_with_layout, example_model_name
    ) -> None:
        """Test layout preparation is skipped if the model already exists."""
        self.plt_mock.Figure.reset_mock()
        tracker_with_layout._prepare_layout(example_model_name, [])
        self.plt_mock.Figure.assert_not_called()

    def test_plot_metric_new_data(
        self,
        sourced_array,
        tracker_with_layout,
        example_model_name,
        example_source_name,
        example_loss_name,
    ) -> None:
        """Test plotting metric with new data."""
        # call plot_metric
        fig, ax = tracker_with_layout._plot_metric(
            example_model_name, example_loss_name, **sourced_array
        )

        # verify the plot was created
        self.mock_ax.plot.assert_called_once()
        self.mock_ax.relim.assert_called_once()
        self.mock_ax.autoscale_view.assert_called_once()
        self.mock_ax.legend.assert_called_once()
        self.mock_fig.canvas.draw.assert_called_once()
        self.mock_fig.canvas.flush_events.assert_called_once()
        assert fig == self.mock_fig
        assert ax == self.mock_ax

    def test_plot_metric_single_point(
        self,
        tracker_with_layout,
        example_model_name,
        example_source_name,
        example_loss_name,
    ) -> None:
        """Test plotting metric with a single data point."""
        # create test data with a single point
        test_data = np.array([[1, 0.8]])
        sourced_array = {example_source_name: test_data}

        # call plot_metric
        _ = tracker_with_layout._plot_metric(
            example_model_name, example_loss_name, **sourced_array
        )

        # verify the scatter plot was used for a single point
        self.mock_ax.scatter.assert_called_once_with(
            test_data[:, 0],
            test_data[:, 1],
            s=200,
            label=example_source_name,
            marker='D',
        )
        self.mock_ax.plot.assert_not_called()

    def test_plot_metric_update_existing_line(
        self,
        sourced_array,
        tracker_with_layout,
        example_model_name,
        example_source_name,
        example_loss_name,
    ) -> None:
        """Test updating existing line data."""
        # mock existing line
        mock_line = self.plt_mock.Line2D(xdata=[1], ydata=[2])
        mock_line.get_label.return_value = example_source_name
        mock_line.set_xdata = mock_line.set_ydata = lambda x: None
        self.mock_ax.get_lines.return_value = [mock_line]
        _ = tracker_with_layout._plot_metric(
            example_model_name, example_loss_name, **sourced_array
        )

        # verify the existing line was updated without creating a new plot
        self.mock_ax.plot.assert_not_called()
        self.mock_ax.scatter.assert_not_called()

    def test_close_all(self, tracker_with_layout, example_model_name) -> None:
        """Test closing all figures when figures exist."""
        # verify figure exists
        assert example_model_name in tracker_with_layout._model_figure

        # close all figures
        tracker_with_layout.close_all()

        # verify the figure was closed and internal storage cleared
        self.plt_mock.close.assert_called_once_with(self.mock_fig)
        assert tracker_with_layout._model_figure == {}

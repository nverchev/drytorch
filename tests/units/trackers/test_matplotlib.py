"""Tests for the "matplotlib" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('matplotlib'):
    pytest.skip('matplotlib not available', allow_module_level=True)

import matplotlib.pyplot as plt
import numpy as np

from drytorch.core import exceptions
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
        mock_fig.clear = mocker.Mock()

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
        self.plt_mock.figure.return_value = mock_fig
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
        assert tracker._source_colors == {}

    @pytest.mark.parametrize('invalid_palette', ['invalid_palette_name', 123])
    def test_initialization_invalid_palette_raises_value_error(
        self, invalid_palette
    ) -> None:
        """Test initialization with invalid palette raises ValueError."""
        with pytest.raises(ValueError):
            MatPlotter(palette=invalid_palette)  # type: ignore[arg-type]

    def test_source_colors_continuous_colormap(self) -> None:
        """Test continuous colormap spreads colors across multiple sources."""
        plotter = MatPlotter(palette='viridis')
        c0 = plotter._get_source_color('source_0')
        c1 = plotter._get_source_color('source_1')
        c2 = plotter._get_source_color('source_2')
        assert c0 != c1
        assert c1 != c2
        assert c0 != c2

    def test_insufficient_discrete_palette_raises_tracker_error(self) -> None:
        """Test exhausting discrete palette colors raises TrackerError."""
        # tab10 has exactly 10 colors
        plotter = MatPlotter(palette='tab10')
        for i in range(10):
            plotter._get_source_color(f'source_{i}')

        with pytest.raises(exceptions.TrackerError) as exc_info:
            plotter._get_source_color('source_10')

        assert 'insufficient' in str(exc_info.value).lower()

    def test_continuous_palette_supports_arbitrary_sources(self) -> None:
        """Test continuous colormaps assign distinct colors beyond 8 sources."""
        plotter = MatPlotter(palette='viridis')
        colors_seen = set()
        for i in range(16):
            c = plotter._get_source_color(f'source_{i}')
            colors_seen.add(c)

        assert len(colors_seen) == 16

    def test_insufficient_continuous_palette_raises_tracker_error(self) -> None:
        """Test exhausting continuous colormap (N=256) raises TrackerError."""
        plotter = MatPlotter(palette='viridis')
        for i in range(plotter._cmap.N):
            plotter._get_source_color(f'source_{i}')

        with pytest.raises(exceptions.TrackerError) as exc_info:
            plotter._get_source_color(f'source_{plotter._cmap.N}')

        assert 'insufficient' in str(exc_info.value).lower()

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
        self, tracker_with_layout, example_model_name, example_loss_name
    ) -> None:
        """Test layout preparation is skipped if metrics already exist."""
        self.plt_mock.figure.reset_mock()
        tracker_with_layout._prepare_layout(
            example_model_name, [example_loss_name]
        )
        self.plt_mock.figure.assert_not_called()

    def test_prepare_layout_rebuilds_when_new_metric_appears(
        self, tracker_with_layout, example_model_name, example_loss_name
    ) -> None:
        """Test layout is cleared and rebuilt when a new metric appears."""
        self.mock_fig.clear.reset_mock()
        self.mock_fig.add_subplot.reset_mock()
        tracker_with_layout._prepare_layout(
            example_model_name, [example_loss_name, 'accuracy']
        )
        self.mock_fig.clear.assert_called_once()
        assert self.mock_fig.add_subplot.call_count == 2

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
        assert fig == self.mock_fig
        assert ax == self.mock_ax

    def test_display_plot(
        self, tracker_with_layout, example_model_name
    ) -> None:
        """Test display_plot redraws canvas once."""
        tracker_with_layout._display_plot(example_model_name, [])
        self.mock_fig.canvas.draw.assert_called_once()
        self.mock_fig.canvas.flush_events.assert_called_once()

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

        # verify single-point plot with diamond marker and consistent color
        expected_color = tracker_with_layout._get_source_color(
            example_source_name
        )
        self.mock_ax.plot.assert_called_once_with(
            test_data[:, 0],
            test_data[:, 1],
            marker='D',
            markersize=10,
            linestyle='None',
            color=expected_color,
            label=example_source_name,
        )
        self.mock_ax.scatter.assert_not_called()

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

    def test_close(self, tracker_with_layout) -> None:
        """Test close() closes all figures and clears internal storage."""
        # Act
        tracker_with_layout.close()

        # Assert
        self.plt_mock.close.assert_called_once_with(self.mock_fig)
        assert tracker_with_layout._model_figure == {}
        assert tracker_with_layout._source_colors == {}

    def test_clean_up_retains_figures(
        self, tracker_with_layout, example_model_name
    ) -> None:
        """Test clean_up() does not close figures so users can inspect them."""
        # Act
        tracker_with_layout.clean_up()

        # Assert
        self.plt_mock.close.assert_not_called()
        assert example_model_name in tracker_with_layout._model_figure


class TestMatPlotterFigureLifecycle:
    """Tests for real figure registration in pyplot manager."""

    def test_figure_lifecycle(self, example_model_name: str) -> None:
        """Test figure registration with pyplot, clean_up, and close."""
        plotter = MatPlotter()
        try:
            plotter._prepare_layout(example_model_name, ['loss'])
            assert len(plt.get_fignums()) == 1

            plotter.clean_up()
            assert len(plt.get_fignums()) == 1

            plotter.close()
            assert len(plt.get_fignums()) == 0
            assert plotter._model_figure == {}
            assert plotter._source_colors == {}
        finally:
            plt.close('all')

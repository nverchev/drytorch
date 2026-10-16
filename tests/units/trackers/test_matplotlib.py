"""Tests for the ``matplotlib`` module."""

import importlib.util

import pytest


if not importlib.util.find_spec('matplotlib'):
    pytest.skip('matplotlib not available', allow_module_level=True)

from matplotlib import figure

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
        return

    @pytest.fixture
    def tracker(
        self, example_model_name, example_loss_name, example_source_name
    ) -> MatPlotter:
        """Set up the instance with metrics in memory."""
        instance = MatPlotter()
        instance.model_dict[example_model_name] = {
            example_source_name: (
                [1, 2, 3],
                {
                    example_loss_name: [0.8, 0.85, 0.9],
                    'Accuracy': [0.5, 0.6, 0.7],
                },
            ),
            'single_point_source': ([3], {'Accuracy': [0.65]}),
        }
        return instance

    def test_create_figures_without_pyplot(
        self, tracker, example_model_name
    ) -> None:
        """The figures are created without pyplot, so they are not rendered."""
        figures = tracker.create_figures(example_model_name)

        assert all(isinstance(fig, figure.Figure) for fig in figures.values())
        self.plt_mock.figure.assert_not_called()
        self.plt_mock.show.assert_not_called()

    def test_create_figures_one_for_each_metric(
        self, tracker, example_model_name, example_loss_name
    ) -> None:
        """Each metric has its own figure, labelled with its name."""
        figures = tracker.create_figures(example_model_name)

        assert set(figures) == {example_loss_name, 'Accuracy'}
        ax = figures['Accuracy'].axes[0]
        assert ax.get_ylabel() == 'Accuracy'
        assert ax.get_title() == example_model_name

    def test_create_figures_single_point(
        self, tracker, example_model_name
    ) -> None:
        """A single point is drawn as a diamond marker."""
        figures = tracker.create_figures(example_model_name)

        lines = {
            line.get_label(): line
            for line in figures['Accuracy'].axes[0].get_lines()
        }
        assert lines['single_point_source'].get_marker() == 'D'
        assert lines['single_point_source'].get_linestyle() == 'None'

    def test_create_figures_same_color_for_a_source(
        self, tracker, example_model_name, example_source_name
    ) -> None:
        """A source has the same color in every figure."""
        figures = tracker.create_figures(example_model_name)

        colors = set()
        for fig in figures.values():
            for line in fig.axes[0].get_lines():
                if line.get_label() == example_source_name:
                    colors.add(line.get_color())

        assert len(colors) == 1

    def test_plot_renders_with_pyplot(
        self, tracker, example_model_name
    ) -> None:
        """Plotting hands every figure to pyplot and shows them together."""
        tracker.plot(example_model_name)

        handed = [call.args[0] for call in self.plt_mock.figure.call_args_list]
        assert len(handed) == 2
        assert all(isinstance(fig, figure.Figure) for fig in handed)
        self.plt_mock.show.assert_called_once()

    def test_plot_grid(
        self, tracker, example_model_name, example_loss_name
    ) -> None:
        """The metrics are subplots of one pyplot figure, shown once."""
        grid = figure.Figure()
        self.plt_mock.figure.return_value = grid

        tracker.plot_grid(example_model_name)

        ylabels = {ax.get_ylabel() for ax in grid.axes}
        assert ylabels == {example_loss_name, 'Accuracy'}
        self.plt_mock.figure.assert_called_once_with()
        self.plt_mock.show.assert_called_once()

    def test_close(self, tracker) -> None:
        """Closing clears the colors of the sources."""
        tracker._get_source_color('source_0')

        tracker.close()

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

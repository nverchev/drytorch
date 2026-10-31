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
        self,
        example_model_name,
        example_loss_name,
        example_source_name,
        start_experiment_mock_event,
    ) -> MatPlotter:
        """Set up the instance with metrics in memory."""
        start_experiment_mock_event.exp_name = 'exp_1'
        instance = MatPlotter()
        instance.notify(start_experiment_mock_event)
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

    def test_create_training_plots_without_pyplot(
        self, tracker, example_model_name
    ) -> None:
        """The figures are created without pyplot, so they are not rendered."""
        figures = tracker.create_training_plots(example_model_name)

        assert all(isinstance(fig, figure.Figure) for fig in figures.values())
        self.plt_mock.figure.assert_not_called()
        self.plt_mock.show.assert_not_called()

    def test_create_training_plots_one_for_each_metric(
        self, tracker, example_model_name, example_loss_name
    ) -> None:
        """Each metric has its own figure, labelled with its name."""
        figures = tracker.create_training_plots(example_model_name)

        assert set(figures) == {example_loss_name, 'Accuracy'}
        ax = figures['Accuracy'].axes[0]
        assert ax.get_ylabel() == 'Accuracy'
        assert ax.get_title() == example_model_name

    def test_create_training_plots_single_point(
        self, tracker, example_model_name
    ) -> None:
        """A single point is drawn as a diamond marker."""
        figures = tracker.create_training_plots(example_model_name)

        lines = {
            line.get_label(): line
            for line in figures['Accuracy'].axes[0].get_lines()
        }
        assert lines['single_point_source'].get_marker() == 'D'
        assert lines['single_point_source'].get_linestyle() == 'None'

    def test_create_training_plots_same_color_for_a_source(
        self, tracker, example_model_name, example_source_name
    ) -> None:
        """A source has the same color in every figure."""
        figures = tracker.create_training_plots(example_model_name)

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

    def test_create_test_plot_one_panel_for_each_metric(
        self, tracker, example_model_name
    ) -> None:
        """Each test metric has its own panel, with tests on the x axis."""
        tracker.test_dict[example_model_name] = {
            'Test': ([1, 2], {'Accuracy': [0.8, 0.9], 'Loss': [0.4, 0.3]}),
            'Test_shifted': ([1], {'Accuracy': [0.7]}),
        }

        test_figure = tracker.create_test_plot(example_model_name)

        assert test_figure is not None
        panels = {ax.get_title(): ax for ax in test_figure.axes}
        assert set(panels) == {'Accuracy', 'Loss'}
        assert len(panels['Accuracy'].patches) == 2
        assert len(panels['Loss'].patches) == 1
        acc_ticks = [t.get_text() for t in panels['Accuracy'].get_xticklabels()]
        assert acc_ticks == ['Test', 'Test_shifted']
        assert len(test_figure.legends) == 1
        legend_labels = [
            t.get_text() for t in test_figure.legends[0].get_texts()
        ]
        assert legend_labels == ['exp_1']
        exp1_color = tracker._get_source_color('exp_1')
        acc_bars = panels['Accuracy'].patches
        assert acc_bars[0].get_facecolor() == exp1_color
        assert acc_bars[1].get_facecolor() == exp1_color

        two_exp_results = {
            'exp_1': {
                'Test': ([2], {'Accuracy': [0.9], 'Loss': [0.3]}),
                'Test_shifted': ([1], {'Accuracy': [0.7]}),
            },
            'exp_2': {
                'Test': ([1], {'Accuracy': [0.85], 'Loss': [0.35]}),
            },
        }
        two_exp_fig = tracker._create_test_plot(
            example_model_name, two_exp_results
        )
        two_panels = {ax.get_title(): ax for ax in two_exp_fig.axes}
        assert len(two_panels['Accuracy'].patches) == 3
        assert len(two_panels['Loss'].patches) == 2
        two_legend = [t.get_text() for t in two_exp_fig.legends[0].get_texts()]
        assert two_legend == ['exp_1', 'exp_2']
        exp2_color = tracker._get_source_color('exp_2')
        assert two_panels['Accuracy'].patches[0].get_facecolor() == exp1_color
        assert two_panels['Accuracy'].patches[1].get_facecolor() == exp1_color
        assert two_panels['Accuracy'].patches[2].get_facecolor() == exp2_color

    def test_plot_grid_outside_run_raises(self, example_model_name) -> None:
        """plot_grid outside a run raises AccessOutsideScopeError."""
        tracker = MatPlotter()
        tracker.model_dict[example_model_name] = {
            'source_0': ([1], {'Loss': [0.5]}),
        }
        with pytest.raises(exceptions.AccessOutsideScopeError):
            tracker.plot_grid(example_model_name)

        tracker_tests_only = MatPlotter()
        tracker_tests_only.test_dict[example_model_name] = {
            'Test': ([1], {'Loss': [0.1]}),
        }
        with pytest.raises(exceptions.AccessOutsideScopeError):
            tracker_tests_only.plot_grid(example_model_name)

        return

    def test_plot_grid_shows_tests(self, tracker, example_model_name) -> None:
        """The test figure is shown with the grid of the curves."""
        tracker.test_dict[example_model_name] = {
            'Test': ([5], {'Accuracy': [0.9]}),
        }
        grid = figure.Figure()
        self.plt_mock.figure.return_value = grid

        tracker.plot_grid(example_model_name)

        handed = [call.args for call in self.plt_mock.figure.call_args_list]
        assert handed[0] == ()
        assert isinstance(handed[1][0], figure.Figure)
        self.plt_mock.show.assert_called_once()

    def test_close(self, tracker) -> None:
        """Closing clears the colors of the sources."""
        tracker._get_source_color('source_0')

        tracker.close()

        assert tracker._source_colors == {}

    def test_folder_name(self) -> None:
        """Test the folder name is 'matplotlib'."""
        assert MatPlotter.folder_name == 'matplotlib'

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

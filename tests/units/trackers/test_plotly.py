"""Tests for the "plotly" module."""

import importlib.util

import pytest


if not importlib.util.find_spec('plotly'):
    pytest.skip('plotly not available', allow_module_level=True)

import plotly.graph_objs as go

from drytorch.trackers.plotly import PlotlyPlotter


class TestPlotlyPlotter:
    """Tests for the ``PlotlyPlotter`` class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test."""
        self.mock_show = mocker.patch.object(go.Figure, 'show')
        return

    @pytest.fixture
    def tracker(
        self, example_model_name, example_loss_name, example_source_name
    ) -> PlotlyPlotter:
        """Set up the instance with metrics in memory."""
        instance = PlotlyPlotter()
        instance.model_dict[example_model_name] = {
            example_source_name: (
                [1, 2, 3],
                {
                    example_loss_name: [0.7, 0.8, 0.85],
                    'Accuracy': [0.5, 0.6, 0.7],
                },
            ),
            'single_point_source': ([3], {'Accuracy': [0.65]}),
        }
        return instance

    def test_create_training_plots_one_for_each_metric(
        self, tracker, example_model_name, example_loss_name
    ) -> None:
        """Each metric has its own figure, labelled with its name."""
        figures = tracker.create_training_plots(example_model_name)

        assert set(figures) == {example_loss_name, 'Accuracy'}
        assert figures['Accuracy'].layout.yaxis.title.text == 'Accuracy'

    def test_create_training_plots_does_not_render(
        self, tracker, example_model_name
    ) -> None:
        """Creating the figures does not render them."""
        tracker.create_training_plots(example_model_name)

        self.mock_show.assert_not_called()

    def test_plot_shows_each_figure(self, tracker, example_model_name) -> None:
        """Plotting shows the figure of each metric."""
        tracker.plot(example_model_name)

        assert self.mock_show.call_count == 2

    def test_create_test_plot_one_panel_for_each_metric(
        self, tracker, example_model_name
    ) -> None:
        """Each test metric has its own panel, with a bar for each test."""
        tracker.test_dict[example_model_name] = {
            'Test': (5, {'Accuracy': 0.9, 'Loss': 0.3}),
            'Test_shifted': (5, {'Accuracy': 0.7}),
        }

        test_figure = tracker.create_test_plot(example_model_name)

        assert test_figure is not None
        bars = list(test_figure.data)
        assert [len(bar.y) for bar in bars] == [2, 1]
        assert all(isinstance(bar, go.Bar) for bar in bars)

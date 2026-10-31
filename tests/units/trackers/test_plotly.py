"""Tests for the "plotly" module."""

import importlib.util
import math

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
        self,
        example_model_name,
        example_loss_name,
        example_source_name,
        start_experiment_mock_event,
    ) -> PlotlyPlotter:
        """Set up the instance with metrics in memory."""
        start_experiment_mock_event.exp_name = 'exp_1'
        instance = PlotlyPlotter()
        instance.notify(start_experiment_mock_event)
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

    def test_folder_name(self) -> None:
        """Test the folder name is 'plotly'."""
        assert PlotlyPlotter.folder_name == 'plotly'

    def test_plot_shows_each_figure(self, tracker, example_model_name) -> None:
        """Plotting shows the figure of each metric."""
        tracker.plot(example_model_name)

        assert self.mock_show.call_count == 2

    def test_create_test_plot_one_panel_for_each_metric(
        self, tracker, example_model_name
    ) -> None:
        """Each test metric has its own panel, with grouped bars."""
        tracker.test_dict[example_model_name] = {
            'Test': ([1, 2], {'Accuracy': [0.8, 0.9], 'Loss': [0.4, 0.3]}),
            'Test_shifted': ([1], {'Accuracy': [0.7]}),
        }

        test_plot = tracker.create_test_plot(example_model_name)

        assert test_plot is not None
        bars = list(test_plot.data)
        assert len(bars) == 2
        assert all(isinstance(bar, go.Bar) for bar in bars)
        assert tuple(bars[0].x) == ('Test', 'Test_shifted')
        assert list(bars[0].y) == [0.9, 0.7]
        assert bars[0].name == 'exp_1'
        assert bars[0].showlegend is True

        assert tuple(bars[1].x) == ('Test',)
        assert list(bars[1].y) == [0.3]
        assert bars[1].name == 'exp_1'
        assert bars[1].showlegend is False

        two_exp_results = {
            'exp_1': {
                'Test': ([2], {'Accuracy': [0.9], 'Loss': [0.3]}),
                'Test_shifted': ([1], {'Accuracy': [0.7]}),
            },
            'exp_2': {
                'Test': ([1], {'Accuracy': [0.85], 'Loss': [0.35]}),
            },
        }
        two_exp_plot = tracker._create_test_plot(
            example_model_name, two_exp_results
        )
        two_bars = list(two_exp_plot.data)
        assert len(two_bars) == 4
        assert tuple(two_bars[0].x) == ('Test', 'Test_shifted')
        assert list(two_bars[0].y) == [0.9, 0.7]
        assert two_bars[0].name == 'exp_1'
        assert two_bars[0].showlegend is True

        assert tuple(two_bars[1].x) == ('Test', 'Test_shifted')
        assert two_bars[1].y[0] == 0.85
        assert math.isnan(two_bars[1].y[1])
        assert two_bars[1].name == 'exp_2'
        assert two_bars[1].showlegend is True

        assert tuple(two_bars[2].x) == ('Test',)
        assert list(two_bars[2].y) == [0.3]
        assert two_bars[2].name == 'exp_1'
        assert two_bars[2].showlegend is False

        assert tuple(two_bars[3].x) == ('Test',)
        assert list(two_bars[3].y) == [0.35]
        assert two_bars[3].name == 'exp_2'
        assert two_bars[3].showlegend is False
        return

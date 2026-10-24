"""Functional tests for PlotlyPlotter."""

import importlib.util
import pathlib

import pytest


if not importlib.util.find_spec('plotly'):
    pytest.skip('plotly not available', allow_module_level=True)

from drytorch.core import log_events
from drytorch.trackers.plotly import PlotlyPlotter


class TestPlotlyPlotterFullCycle:
    """Complete ``PlotlyPlotter`` session."""

    def test_figures_of_the_session(
        self,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """A session has a figure for each metric."""
        plotter = PlotlyPlotter()
        for event in plotting_workflow:
            plotter.notify(event)

        figures = plotter.create_training_plots(example_model_name)
        plotter.close()

        assert len(figures) == 3

    def test_save_plots_of_the_run(
        self,
        tmp_path: pathlib.Path,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """Each metric and the tests are saved as files in the run's folder."""
        plotter = PlotlyPlotter()
        for event in plotting_workflow:
            plotter.notify(event)

        saved = {
            path.stem
            for path in tmp_path.glob(f'plots/**/{example_model_name}/*.html')
        }
        assert len(saved) == 4
        assert 'test_metrics' in saved

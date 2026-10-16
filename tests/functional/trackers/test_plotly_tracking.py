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

        figures = plotter.create_figures(example_model_name)
        plotter.close()

        assert len(figures) == 3

    def test_save_figures_of_the_run(
        self,
        tmp_path: pathlib.Path,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """Each figure is saved as a file in the folder of the run."""
        plotter = PlotlyPlotter()
        *events, stop_event = plotting_workflow
        for event in events:
            plotter.notify(event)

        plotter.save_figures(example_model_name)
        plotter.notify(stop_event)

        saved = list(tmp_path.glob(f'plots/**/{example_model_name}/*.html'))
        assert len(saved) == 3

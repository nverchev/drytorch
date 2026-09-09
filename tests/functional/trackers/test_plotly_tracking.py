"""Functional tests for PlotlyPlotter."""

import importlib.util
import os

import pytest


if not importlib.util.find_spec('plotly'):
    pytest.skip('plotly not available', allow_module_level=True)

from drytorch.core import log_events
from drytorch.trackers.plotly import PlotlyPlotter


class TestPlotlyPlotterFullCycle:
    """Manual verification for PlotlyPlotter."""

    def test_live_plot(
        self,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """Render plotly figures interactively when LIVE_PLOT is set."""
        plotter = PlotlyPlotter()

        for event in plotting_workflow:
            plotter.notify(event)

        figures = plotter.plot(example_model_name)
        assert len(figures) == 3

        if 'LIVE_PLOT' in os.environ:
            for fig in figures:
                fig.show()

        plotter.clean_up()
        plotter.close()

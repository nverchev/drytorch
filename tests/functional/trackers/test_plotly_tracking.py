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

    @pytest.mark.skipif(
        'LIVE_PLOT' not in os.environ,
        reason='opens figure windows; run manually with LIVE_PLOT set',
    )
    def test_live_plot(
        self,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """Render plotly figures interactively when LIVE_PLOT is set."""
        # Prepare
        plotter = PlotlyPlotter()

        # Trigger
        for event in plotting_workflow:
            plotter.notify(event)
        figures = plotter.plot(example_model_name)
        plotter.clean_up()
        plotter.close()

        # Assert
        assert len(figures) == 3

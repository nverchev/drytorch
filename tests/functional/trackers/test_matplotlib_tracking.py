"""Functional tests for MatPlotter."""

import importlib.util
import os

import pytest


if not importlib.util.find_spec('matplotlib'):
    pytest.skip('matplotlib not available', allow_module_level=True)

import matplotlib.pyplot as plt

from drytorch.core import log_events
from drytorch.trackers.matplotlib import MatPlotter


class TestMatPlotterFullCycle:
    """Complete MatPlotter session and manual verification."""

    @pytest.mark.skipif(
        'LIVE_PLOT' not in os.environ,
        reason='opens figure windows; run manually with LIVE_PLOT set',
    )
    def test_live_plotting_lifecycle(
        self,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """Verify multi-epoch, multi-metric training with live plots."""
        pause_epoch = 0.5
        pause_final = 2.0
        plotter = MatPlotter()

        for event in plotting_workflow:
            plotter.notify(event)
            if isinstance(
                event, (log_events.EndEpochEvent, log_events.EndTestEvent)
            ):
                plt.pause(pause_epoch)

        # 4. clean_up() should keep figures open for user inspection
        plotter.clean_up()
        assert len(plt.get_fignums()) == 1
        plt.pause(pause_final)

        # 5. close() should close all figures and clear state
        plotter.close()
        assert len(plt.get_fignums()) == 0
        assert plotter._model_figure == {}
        assert plotter._source_colors == {}

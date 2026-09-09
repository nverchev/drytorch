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
        self, example_model_name: str, example_source_name: str
    ) -> None:
        """Render plotly figures interactively when LIVE_PLOT is set."""
        model_name = example_model_name
        train_source = example_source_name
        val_source = 'val'
        test_source = 'test'
        plotter = PlotlyPlotter()

        # Simulated training metrics over 10 epochs
        train_loss = [1.8, 1.4, 1.1, 0.85, 0.65, 0.50, 0.40, 0.32, 0.25, 0.20]
        train_acc = [
            0.35,
            0.50,
            0.62,
            0.70,
            0.78,
            0.84,
            0.88,
            0.91,
            0.94,
            0.96,
        ]

        # Validation metrics over 10 epochs
        val_loss = [1.5, 1.25, 1.0, 0.85, 0.75, 0.70, 0.68, 0.67, 0.66, 0.65]
        val_acc = [0.45, 0.55, 0.65, 0.72, 0.77, 0.80, 0.82, 0.83, 0.84, 0.85]
        val_f1 = [0.42, 0.53, 0.63, 0.70, 0.75, 0.78, 0.80, 0.81, 0.82, 0.83]

        for epoch in range(1, 11):
            plotter.notify(
                log_events.MetricEvent(
                    model_name=model_name,
                    source_name=train_source,
                    epoch=epoch,
                    metrics={
                        'loss': train_loss[epoch - 1],
                        'accuracy': train_acc[epoch - 1],
                    },
                )
            )
            plotter.notify(
                log_events.MetricEvent(
                    model_name=model_name,
                    source_name=val_source,
                    epoch=epoch,
                    metrics={
                        'loss': val_loss[epoch - 1],
                        'accuracy': val_acc[epoch - 1],
                        'f1_score': val_f1[epoch - 1],
                    },
                )
            )
            plotter.notify(
                log_events.EndEpochEvent(
                    source_name=train_source,
                    model_name=model_name,
                    epoch=epoch,
                )
            )

        # Test evaluation at the end (test split: single point per metric)
        plotter.notify(
            log_events.MetricEvent(
                model_name=model_name,
                source_name=test_source,
                epoch=10,
                metrics={'loss': 0.69, 'accuracy': 0.83, 'f1_score': 0.81},
            )
        )
        plotter.notify(
            log_events.EndTestEvent(
                source_name=test_source, model_name=model_name
            )
        )

        figures = plotter.plot(model_name)
        assert len(figures) == 3

        if 'LIVE_PLOT' in os.environ:
            for fig in figures:
                fig.show()

        plotter.clean_up()
        plotter.close()

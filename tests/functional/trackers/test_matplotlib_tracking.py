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

    def test_live_plotting_lifecycle(
        self, example_model_name: str, example_source_name: str
    ) -> None:
        """Verify multi-epoch, multi-metric training with live plots."""
        is_live = 'LIVE_PLOT' in os.environ
        pause_epoch = 0.5 if is_live else 0.001
        pause_final = 2.0 if is_live else 0.001
        model_name = example_model_name
        train_source = example_source_name
        val_source = 'val'
        test_source = 'test'
        plotter = MatPlotter()

        # Simulated training metrics over 10 epochs
        train_loss = [1.8, 1.4, 1.1, 0.85, 0.65, 0.50, 0.40, 0.32, 0.25, 0.20]
        train_acc = [0.35, 0.50, 0.62, 0.70, 0.78, 0.84, 0.88, 0.91, 0.94, 0.96]

        # Validation metrics over 10 epochs
        val_loss = [1.5, 1.25, 1.0, 0.85, 0.75, 0.70, 0.68, 0.67, 0.66, 0.65]
        val_acc = [0.45, 0.55, 0.65, 0.72, 0.77, 0.80, 0.82, 0.83, 0.84, 0.85]
        val_f1 = [0.42, 0.53, 0.63, 0.70, 0.75, 0.78, 0.80, 0.81, 0.82, 0.83]

        # 10 Epochs: 3 metrics total (accuracy, f1_score, loss) across sources
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
            # Pause so plots can be visually inspected
            plt.pause(pause_epoch)

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

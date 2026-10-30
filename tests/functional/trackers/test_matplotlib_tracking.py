"""Functional tests for MatPlotter."""

import dataclasses
import importlib.util
import pathlib

import pytest


if not importlib.util.find_spec('matplotlib'):
    pytest.skip('matplotlib not available', allow_module_level=True)

from drytorch.core import log_events
from drytorch.trackers.csv import CSVDumper
from drytorch.trackers.matplotlib import MatPlotter


class TestMatPlotterFullCycle:
    """Complete ``MatPlotter`` session."""

    def test_figures_of_the_session(
        self,
        tmp_path: pathlib.Path,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """A session has a figure for each metric, which saves to a file."""
        plotter = MatPlotter()
        for event in plotting_workflow:
            plotter.notify(event)

        figures = plotter.create_training_plots(example_model_name)
        for metric_name, fig in figures.items():
            fig.savefig(tmp_path / f'{metric_name}.png')

        plotter.close()

        assert len(figures) == 3
        assert len(list(tmp_path.glob('*.png'))) == 3

    def test_save_plots_of_the_run(
        self,
        tmp_path: pathlib.Path,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """Each metric and the tests are saved as files in the run's folder."""
        plotter = MatPlotter()
        for event in plotting_workflow:
            plotter.notify(event)

        saved = {
            path.stem
            for path in tmp_path.glob(
                f'matplotlib/**/{example_model_name}/*.png'
            )
        }
        assert len(saved) == 4
        assert 'test_metrics' in saved

    def test_resumed_run_with_csv_loader_saves_earlier_test_plot(
        self,
        tmp_path: pathlib.Path,
        example_model_name: str,
        start_experiment_event: log_events.StartExperimentEvent,
        stop_experiment_event: log_events.StopExperimentEvent,
    ) -> None:
        """A resumed run with CSV loader still saves earlier test plot."""
        csv_tracker = CSVDumper(par_dir=tmp_path)
        csv_tracker.notify(start_experiment_event)

        train_event = log_events.MetricEvent(
            model_name=example_model_name,
            source_name='train',
            epoch=1,
            metrics={'loss': 0.5, 'accuracy': 0.8},
        )
        val_event = log_events.MetricEvent(
            model_name=example_model_name,
            source_name='val',
            epoch=1,
            metrics={'loss': 0.45, 'accuracy': 0.82},
        )
        test_event = log_events.TestMetricEvent(
            model_name=example_model_name,
            source_name='test',
            epoch=1,
            metrics={'loss': 0.4, 'accuracy': 0.85},
        )
        csv_tracker.notify(train_event)
        csv_tracker.notify(val_event)
        csv_tracker.notify(test_event)
        csv_tracker.notify(stop_experiment_event)

        resumed_start = dataclasses.replace(
            start_experiment_event, resumed=True
        )
        resumed_csv = CSVDumper(par_dir=tmp_path)
        plotter = MatPlotter(metric_loader=resumed_csv, par_dir=tmp_path)
        plotter.notify(resumed_start)
        resumed_csv.notify(resumed_start)

        plotter.save_plots(example_model_name)
        plotter.notify(stop_experiment_event)
        resumed_csv.notify(stop_experiment_event)

        saved = {
            path.stem
            for path in tmp_path.glob(
                f'matplotlib/**/{example_model_name}/*.png'
            )
        }
        assert 'test_metrics' in saved

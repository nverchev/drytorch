"""Functional tests for PlotlyPlotter."""

import dataclasses
import importlib.util
import pathlib

import pytest


if not importlib.util.find_spec('plotly'):
    pytest.skip('plotly not available', allow_module_level=True)

from drytorch.core import log_events
from drytorch.trackers.csv import CSVDumper
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
        for event in plotting_workflow[:-1]:
            plotter.notify(event)

        figures = plotter.create_training_plots(example_model_name)
        plotter.notify(plotting_workflow[-1])
        plotter.close()

        assert len(figures) == 3

    def test_save_plots_of_the_run(
        self,
        tmp_path: pathlib.Path,
        example_model_name: str,
        plotting_workflow: tuple[log_events.Event, ...],
    ) -> None:
        """The curves are saved in the run folder, and tests in test_metrics."""
        plotter = PlotlyPlotter()
        for event in plotting_workflow:
            plotter.notify(event)

        saved = {
            path.stem
            for path in tmp_path.glob(f'plotly/**/{example_model_name}/*.html')
        }
        assert len(saved) == 3
        test_plot = (
            tmp_path / 'plotly' / 'test_metrics' / f'{example_model_name}.html'
        )
        assert test_plot.is_file()
        return

    def test_multiple_experiments_save_shared_test_plot(
        self,
        tmp_path: pathlib.Path,
        example_model_name: str,
        start_experiment_event: log_events.StartExperimentEvent,
        stop_experiment_event: log_events.StopExperimentEvent,
    ) -> None:
        """Two experiments with CSV loader produce one shared test plot file."""
        csv_tracker = CSVDumper(par_dir=tmp_path)
        plotter = PlotlyPlotter(metric_loader=csv_tracker, par_dir=tmp_path)

        start_1 = dataclasses.replace(
            start_experiment_event, exp_name='exp_1', run_id='run_1'
        )
        csv_tracker.notify(start_1)
        plotter.notify(start_1)
        test_event_1 = log_events.TestMetricEvent(
            model_name=example_model_name,
            source_name='test',
            epoch=1,
            metrics={'accuracy': 0.8},
        )
        csv_tracker.notify(test_event_1)
        plotter.notify(test_event_1)
        plotter.notify(stop_experiment_event)
        csv_tracker.notify(stop_experiment_event)

        start_2 = dataclasses.replace(
            start_experiment_event, exp_name='exp_2', run_id='run_1'
        )
        csv_tracker.notify(start_2)
        plotter.notify(start_2)
        test_event_2 = log_events.TestMetricEvent(
            model_name=example_model_name,
            source_name='test',
            epoch=1,
            metrics={'accuracy': 0.9},
        )
        csv_tracker.notify(test_event_2)
        plotter.notify(test_event_2)
        plotter.notify(stop_experiment_event)
        csv_tracker.notify(stop_experiment_event)

        shared_plot_file = (
            tmp_path / 'plotly' / 'test_metrics' / f'{example_model_name}.html'
        )
        assert shared_plot_file.is_file()
        return

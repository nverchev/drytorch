"""Functional tests for MatPlotter."""

import importlib.util
import pathlib

import pytest


if not importlib.util.find_spec('matplotlib'):
    pytest.skip('matplotlib not available', allow_module_level=True)

from drytorch.core import log_events
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
        *events, stop_event = plotting_workflow
        for event in events:
            plotter.notify(event)

        plotter.save_plots(example_model_name)
        plotter.notify(stop_event)

        saved = {
            path.stem
            for path in tmp_path.glob(f'plots/**/{example_model_name}/*.png')
        }
        assert len(saved) == 4
        assert 'tests' in saved

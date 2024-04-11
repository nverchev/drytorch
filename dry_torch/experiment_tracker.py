import warnings
from typing import Any, Literal

import pandas as pd

from dry_torch.plotters import GetPlotterProtocol, plotter_closure, Plotter, Backend
from dry_torch.recursive_ops import struc_repr
from dry_torch.data_manager import DatasetName

# TypedDict currently not able to correctly infer type of keys and values.
LogMetrics = dict[DatasetName, pd.DataFrame]


class ExperimentTracker:
    """
    Keep track of experiment and provide a plotting interface.

    Args:
         exp_name: The name of the experiment.
         kwargs: Settings, options, variables, anything that impacts or defines the experiment.

    Methods:
        extract_metadata: attempt to fully document the experiment.
        plot_learning_curves: plot the learning curves using an available backend.
    """
    max_length_string_repr: int = 10

    def __init__(self, exp_name, **kwargs) -> None:
        self.exp_name: str = exp_name
        self.metadata: dict[str, Any] = self.extract_metadata(**kwargs)

        self.log: LogMetrics = {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
        self.get_plotter: GetPlotterProtocol = plotter_closure()

    def extract_metadata(self, **kwargs) -> dict[str, Any]:

        # tries to get the most informative representation of the metadata.
        try:
            kwargs = {k: struc_repr(v, max_length=self.max_length_string_repr) for k, v in kwargs.items()}
        except RecursionError:
            warnings.warn(f'Could not provide a hierarchical representation of the metadata.')
            kwargs = {}
        model_architecture = {'model': kwargs['model']}  # JSON files do not allow strings on multiple lines
        model_settings: dict[str, Any] = getattr(kwargs['model'], 'metadata', {})
        return model_architecture | model_settings | kwargs

    def plot_learning_curves(self, loss_or_metric: str = 'Criterion', start: int = 0,
                             title: str = 'Learning Curves', lib: Backend = 'auto') -> None:
        """
        This method plots the learning curves using either plotly or visdom as backends

        Args:
            loss_or_metric: the loss_fun or the metric to visualize.
            start: the epoch from where you want to display the curve.
            title: the name of the window (and title) of the plot in the visdom interface.
            lib: which library to use between visdom and plotly. 'auto' selects plotly if the visdom connection failed.
        """
        if self.log['train'].empty:
            warnings.warn('Plotting learning curves is not possible because data is missing.')
            return
        plotter: Plotter = self.get_plotter(backend=lib, env=self.exp_name)
        plotter.plot(self.log['train'], self.log['val'], loss_or_metric, start, title)
        return

    def __str__(self):
        return self.exp_name

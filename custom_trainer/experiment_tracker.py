import warnings
from typing import Any, Literal

import pandas as pd

from custom_trainer.plotters import GetPlotterProtocol, plotter_backend, Plotter
from custom_trainer.recursive_ops import struc_repr
from custom_trainer.data_manager import DatasetName

# TypedDict currently not able to correctly infer type of keys and values.
LogMetrics = dict[DatasetName, pd.DataFrame]


class ExperimentTracker(object):
    """
                exp_name: name used for the folder containing the checkpoints and the metadata

    """
    max_length_string_repr: int = 10

    def __init__(self, **kwargs) -> None:

        # tries to get the most informative representation of the settings.
        try:
            kwargs = {k: struc_repr(v, max_length=self.max_length_string_repr) for k, v in kwargs.items()}
        except RecursionError:
            warnings.warn(f'Could not provide a hierarchical representation of the settings.')
            kwargs = {}
        model_architecture = {'model': kwargs['model']}  # JSON files do not allow strings on multiple lines
        model_settings: dict[str, Any] = getattr(kwargs['model'], 'settings', {})
        self.settings: dict[str, Any] = model_architecture | model_settings | kwargs

        self.exp_name: str = kwargs['exp_name']
        self.log: LogMetrics = {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
        self.get_plotter: GetPlotterProtocol = plotter_backend()

    def plot_learning_curves(self, loss_or_metric: str = 'Criterion', start: int = 0,
                             title: str = 'Learning Curves', lib: Literal['visdom', 'plotly', 'auto'] = 'auto') -> None:
        """
        This method plots the learning curves using either plotly or visdom as backends

        Args:
            loss_or_metric: the loss or the metric to visualize
            start: the epoch from where you want to display the curve
            title: the name of the window (and title) of the plot in the visdom interface
            lib: which library to use between visdom and plotly. 'auto' selects plotly if the visdom connection failed.
        """
        if self.log['train'].empty:
            warnings.warn('Plotting learning curves is not possible because data is missing.')
            return
        plotter: Plotter = self.get_plotter(backend=lib, env=self.exp_name)
        plotter.plot(self.log['train'], self.log['val'], loss_or_metric, start, title)
        return

    def overwrite(self, df_name, df):
        self.__setattr__(df_name, df)

    def __str__(self):
        return self.exp_name

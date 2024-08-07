"""
Module for plotting backends and the Plotting class.

It tries to import the supported backend libraries for plotting and create
a plotter for each, while the Plotter class functions as an interface.
"""

from abc import ABCMeta, abstractmethod
import os
from types import ModuleType
from typing import Callable, Literal
import warnings

import pandas as pd
from dry_torch import protocols as p
from dry_torch import tracking
from dry_torch import exceptions
from dry_torch import descriptors

_Backend = Literal['visdom', 'plotly', 'auto', 'none']


class BasePlotter(p.PlotterProtocol, metaclass=ABCMeta):
    """
    Abstract base class for plotting learning curves.

    Args:
        model_name: the name of the model. Default to ''.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name

    def plot(self,
             train_log: pd.DataFrame,
             val_log: pd.DataFrame,
             metric_name: str = 'Criterion',
             title: str = 'Learning Curves') -> None:
        """
        Plots the learning curves.

        Args:
            train_log: DataFrame with the logged training metrics.
            val_log: DataFrame with the logged validation metrics.
            metric_name: the metric to visualize. Defaults to 'Criterion'.
            title: the title of the plot. Defaults to 'Learning Curves'.
        """
        self._plot(train_log, val_log, metric_name, title)

    @abstractmethod
    def _plot(self,
              train_log: pd.DataFrame,
              val_log: pd.DataFrame,
              loss_or_metric: str,
              title: str) -> None:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__ + f' for {self.model_name}.'


class NoPlotter(BasePlotter):
    """
    Fallback class when no backend is available. Plotting is silently ignored.
    """

    def _plot(self,
              train_log: pd.DataFrame,
              val_log: pd.DataFrame,
              loss_or_metric: str,
              title: str) -> None:
        pass


try:
    import visdom

    VISDOM_AVAILABLE: bool = True

except ImportError:
    VISDOM_AVAILABLE = False
    visdom = ModuleType('Unreachable module.')

else:

    class VisdomPlotter(BasePlotter):
        """
        Initialize a visdom environment uses it as backend.

        Args:
            model_name: name given to the visdom environment.
        """

        def __init__(self, model_name: str) -> None:
            super().__init__(model_name)
            self.model_name = model_name
            self.vis = visdom.Visdom(env=model_name, raise_exceptions=True)

        def check_connection(self) -> bool:
            return self.vis.check_connection()

        def _plot(self,
                  train_log: pd.DataFrame,
                  val_log: pd.DataFrame,
                  metric: str,
                  title: str) -> None:
            train_log_metric: pd.Series[float] = train_log.get(metric,
                                                               pd.Series(index=train_log.index, name='Size'))
            layout = dict(xlabel='Epoch',
                          ylabel=metric,
                          title=title,
                          update='replace',
                          showlegend=True)
            self.vis.line(X=train_log['Epoch'],
                          Y=train_log_metric,
                          win=title,
                          opts=layout,
                          name='Training')
            for source, source_log in val_log.groupby('Source'):
                val_log_metric: pd.Series[float] = source_log[metric]
                self.vis.line(X=source_log['Epoch'],
                              Y=val_log_metric,
                              win=title,
                              opts=layout,
                              update='append',
                              name=str(source))
            return

try:
    import plotly.express as px  # type: ignore

    PLOTLY_AVAILABLE: bool = True

except ImportError:
    PLOTLY_AVAILABLE = False
    px = ModuleType('Unreachable module.')

else:
    class PlotlyPlotter(BasePlotter):
        def _plot(self,
                  train_log: pd.DataFrame,
                  val_log: pd.DataFrame,
                  loss_or_metric: str,
                  title: str) -> None:
            train_log = train_log.copy()
            train_log['Dataset'] = "Training"
            val_log = val_log.copy()
            val_log['Dataset'] = val_log['Source']
            # noinspection PyUnreachableCode
            log = pd.concat((train_log, val_log))
            fig = px.line(log,
                          x="Epoch",
                          y=loss_or_metric,
                          color="Source",
                          title=title)
            fig.show()
            return


def plotter_closure(model_name: str,
                    backend: _Backend = 'auto') -> Callable[[], BasePlotter]:
    """
    Closure that caches the last used plotter.

    Args:
        model_name: the name of the model.
        backend:
                auto:
                    from a Jupiter notebook use plotly ot Raise ImportError.
                    otherwise:
                        if visdom is installed use visdom else plotly.
                        if neither are installed return NoPlotter.
                plotly: plotly backend. Raise ImportError if not installed.
                visdom: visdom backend. Raise ImportError if not installed.

    Return:
        A Callable returning the plotter.
    """

    plotter: BasePlotter = NoPlotter('')

    def get_plotter(_backend: _Backend = backend) -> BasePlotter:
        nonlocal plotter
        jupyter = os.path.basename(os.environ['_']) == 'jupyter'
        # True when calling from a jupyter notebook
        if _backend == 'auto':
            if jupyter:
                return get_plotter('plotly')

            try:
                return get_plotter('visdom')
            except ImportError:
                pass

            try:
                return get_plotter('plotly')
            except ImportError:
                return get_plotter('none')

        if _backend == 'plotly':
            if not PLOTLY_AVAILABLE:
                raise exceptions.LibraryNotAvailableError('plotly')

            if not isinstance(plotter, PlotlyPlotter):
                plotter = PlotlyPlotter(model_name)

            return plotter

        if _backend == 'visdom':
            if not VISDOM_AVAILABLE:
                raise exceptions.LibraryNotAvailableError('visdom')

            if isinstance(plotter, VisdomPlotter):
                if plotter.check_connection():
                    return plotter

            try:
                plotter = VisdomPlotter(model_name)
                return plotter
            except ConnectionError:
                warnings.warn(exceptions.VisdomConnectionWarning())
                return get_plotter('none')

        if _backend == 'none':
            if not isinstance(plotter, NoPlotter):
                plotter = NoPlotter(model_name)

            return plotter

        raise exceptions.LibraryNotSupportedError(backend)

    return get_plotter


class Plotter:
    """
    Class providing a plotting interface.

    Args:
         model_name: the name of the model.
         lib: the backend to use for plotting.
    """

    def __init__(self, model_name: str, lib: _Backend = 'auto') -> None:
        self.model_name = model_name
        env = tracking.Experiment.current().name
        self.get_plotter = plotter_closure(env, lib)

    def plot_learning_curves(self,
                             metric_name: str = 'Criterion',
                             start: int = 0,
                             title: str = 'Learning Curves') -> None:
        """
        Plots the learning curves.

        Args:
            metric_name: the metric to visualize.
            start: the starting epoch for the plot.
            title: the title of the plot.
        """
        log = tracking.Experiment.current().tracker[self.model_name].log

        self._plot(log[descriptors.Split.TRAIN],
                   log[descriptors.Split.VAL],
                   metric_name,
                   start,
                   title)
        return

    def _plot(self,
              train_log: pd.DataFrame,
              val_log: pd.DataFrame,
              metric_name: str = 'Criterion',
              start: int = 0,
              title: str = 'Learning Curves',
              ) -> None:
        plotter = self.get_plotter()
        train_log = train_log[train_log['Epoch'] >= start]
        val_log = val_log[val_log['Epoch'] >= start]
        plotter.plot(train_log,
                     val_log,
                     metric_name,
                     title)
        return

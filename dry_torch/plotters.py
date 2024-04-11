import os
from types import ModuleType
from abc import ABCMeta, abstractmethod
from typing import Literal, Protocol
import warnings
from textwrap import dedent

import pandas as pd
from .doc_utils import add_docstring

Backend = Literal['visdom', 'plotly', 'auto', 'none']

plot_docstring: str = dedent("""
        Plot the learning curves.

        Args:
            train_log: DataFrame with the loss_fun and metrics_fun calculated during training on the training dataset.
            val_log: Dataframe with the loss_fun and metrics_fun calculated during training on the validation dataset.
            loss_or_metric: the metric to visualize. Defaults to 'Criterion'.
            start: the epoch from where you want to display the curve. Defaults to 0.
            title: the name of the window of the plot in the visdom interface. Defaults to 'Learning Curves'.
        """)


class Plotter(metaclass=ABCMeta):
    """
    Abstract base class for plotting learning curves.

    Args:
        env: the name of the environment (useful to some backends). Default to ''.
    """

    def __init__(self, env: str = '') -> None:
        self.env = env

    @abstractmethod
    @add_docstring(plot_docstring)
    def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
             loss_or_metric: str = 'Criterion', start: int = 0,
             title: str = 'Learning Curves') -> None:
        ...


class NoPlotter(Plotter):
    """
    Fallback class when no backend is available. Plotting is silently ignored.
    """

    def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
             loss_or_metric: str = 'Criterion', start: int = 0,
             title: str = 'Learning Curves') -> None:
        pass


try:
    import visdom

    VISDOM_AVAILABLE: bool = True


    class VisdomPlotter(Plotter):
        """
        Initialize a visdom environment with the specified environment name and uses it as backend.

        Args:
            env: The name of the visdom environment. Default to ''.
        """

        def __init__(self, env: str = '') -> None:
            super().__init__(env)
            self.env = env
            self.vis: visdom.Visdom = visdom.Visdom(env=env, raise_exceptions=True)

        def check_connection(self) -> bool:
            return self.vis.check_connection()

        @add_docstring(plot_docstring)
        def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
                 loss_or_metric: str = 'Criterion', start: int = 0,
                 title: str = 'Learning Curves') -> None:
            train_log_metric: pd.Series[float] = train_log[train_log.index > start][loss_or_metric]
            val_log_metric: pd.Series[float] = val_log[val_log.index >= start][loss_or_metric]
            layout = dict(xlabel='Epoch', ylabel=loss_or_metric, title=title, update='replace', showlegend=True)
            self.vis.line(X=train_log_metric.index, Y=train_log_metric, win=title, opts=layout, name='Training')
            if not val_log.empty:
                self.vis.line(X=val_log_metric.index, Y=val_log_metric, win=title, opts=layout, update='append',
                              name='Validation')
            return


except ImportError:
    VISDOM_AVAILABLE = False
    visdom = ModuleType('Unreachable module')

try:
    import plotly.express as px  # type: ignore

    PLOTLY_AVAILABLE: bool = True


    class PlotlyPlotter(Plotter):
        @add_docstring(plot_docstring)
        def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
                 loss_or_metric: str = 'Criterion', start: int = 0,
                 title: str = 'Learning Curves') -> None:
            train_log = train_log.copy()
            train_log['Dataset'] = "Training"
            val_log = val_log.copy()
            val_log['Dataset'] = "Validation"
            log = pd.concat([train_log, val_log])
            log = log[log.index >= start].reset_index().rename(columns={'index': 'Epoch'})
            fig = px.line(log, x="Epoch", y=loss_or_metric, color="Dataset", title=title)  # type: ignore
            fig.show()
            return


except ImportError:
    PLOTLY_AVAILABLE = False
    px = ModuleType('Unreachable module')


class GetPlotterProtocol(Protocol):
    """
    This protocol is a Callable with named parameters.
    """

    def __call__(self, backend: Backend, env: str) -> Plotter:
        ...


def plotter_closure() -> GetPlotterProtocol:
    """
    Cache the last used plotter.

    Return:
        A function that returns a new plotter backend or the last used plotter.
    """

    plotter: Plotter = NoPlotter('')

    def get_plotter(backend: Backend = 'auto', env: str = '') -> Plotter:
        """
        Returns a plotter object based on the specified backend.

        Args:
            backend:
                auto (default):
                    from a Jupiter notebook: use plotly and return ImportError if plotly is not installed.
                    otherwise: if visdom is installed use visdom else plotly. If neither are installed return NoPlotter.
                plotly: plotly backend. Return ImportError if plotly is not installed.
                visdom: visdom backend. Return ImportError if visdom is not installed.
            env: The optional environment name for the backend. Default ''.

        Return:
            A plotter backend based on the specified backend and environment.

        Raise:
            ImportError if the specified backend is not available.
            warning if the visdom connection is refused by the server.
        """
        nonlocal plotter
        jupyter = os.path.basename(os.environ['_']) == 'jupyter'  # True when calling from a jupyter notebook
        if backend == 'auto':
            if jupyter:
                return get_plotter('plotly', '')
            try:
                return get_plotter('visdom', env)
            except ImportError:
                pass
            try:
                return get_plotter('plotly', '')
            except ImportError:
                return get_plotter('none', '')

        if backend == 'plotly':
            if not PLOTLY_AVAILABLE:
                raise ImportError(f'Library plotly not installed.')

            if not isinstance(plotter, PlotlyPlotter):
                plotter = PlotlyPlotter()

            return plotter

        if backend == 'visdom':
            if not VISDOM_AVAILABLE:
                raise ImportError(f'Library visdom not installed.')

            if isinstance(plotter, VisdomPlotter):
                if plotter.check_connection():
                    return plotter

            try:
                plotter = VisdomPlotter(env)
                return plotter
            except ConnectionError:
                warnings.warn('Visdom connection refused by server.')
                return get_plotter('none', '')

        if backend == 'none':
            if not isinstance(plotter, NoPlotter):
                plotter = NoPlotter()

            return plotter

        raise ValueError(f'Library {backend} not supported.')

    return get_plotter

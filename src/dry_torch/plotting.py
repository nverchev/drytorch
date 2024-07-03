import os
from types import ModuleType
from abc import ABCMeta, abstractmethod
from typing import Literal, Protocol
import warnings
import pandas as pd
from pandas import DataFrame
from dry_torch import data_types
from dry_torch import tracking
from dry_torch import exceptions

Backend = Literal['visdom', 'plotly', 'auto', 'none']


class BackendPlotter(metaclass=ABCMeta):
    """
    Abstract base class for plotting learning curves.

    Args:
        env: the model_name of the environment (useful to some backends).
        Default to ''.
    """

    def __init__(self, env: str = '') -> None:
        self.env: str = env

    def plot(self,
             train_log: DataFrame,
             val_log: DataFrame,
             loss_or_metric: str = 'Criterion',
             start: int = 0,
             title: str = 'Learning Curves') -> None:
        """
        Plot the learning curves.

        Args:
            train_log: DataFrame with the loss_fun and metrics_fun calculated
            during training on the training dataset.
            val_log: Dataframe with the loss_fun and metrics_fun calculated
            during training on the validation dataset.
            loss_or_metric: the metric to visualize. Defaults to 'Criterion'.
            start: the epoch from where you want to display the curve.
             Defaults to 0.
            title: the model_name of the window of the plot in the visdom interface.
            Defaults to 'Learning Curves'.
        """
        self._plot(train_log, val_log, loss_or_metric, start, title)

    @abstractmethod
    def _plot(self,
              train_log: DataFrame,
              val_log: DataFrame,
              loss_or_metric: str,
              start: int,
              title: str) -> None:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(environment={self.env})'


class NoPlotter(BackendPlotter):
    """
    Fallback class when no backend is available. Plotting is silently ignored.
    """

    def _plot(self,
              train_log: DataFrame,
              val_log: DataFrame,
              loss_or_metric: str,
              start: int,
              title: str) -> None:
        pass


try:
    import visdom

    VISDOM_AVAILABLE: bool = True


    class VisdomPlotter(BackendPlotter):
        """
        Initialize a visdom environment with the specified environment model_name
         and uses it as backend.

        Args:
            env: The model_name of the visdom environment. Default to ''.
        """

        def __init__(self, env: str = '') -> None:
            super().__init__(env)
            self.env = env
            self.vis = visdom.Visdom(env=env, raise_exceptions=True)

        def check_connection(self) -> bool:
            return self.vis.check_connection()

        def _plot(self,
                  train_log: DataFrame,
                  val_log: DataFrame,
                  loss_or_metric: str,
                  start: int,
                  title: str) -> None:
            train_log_metric: pd.Series[float] = train_log[loss_or_metric]
            val_log_metric: pd.Series[float] = val_log[loss_or_metric]
            train_log_metric = train_log_metric[train_log.index >= start]
            val_log_metric = val_log_metric[val_log.index >= start]

            layout = dict(xlabel='Epoch',
                          ylabel=loss_or_metric,
                          title=title,
                          update='replace',
                          showlegend=True)
            self.vis.line(X=train_log_metric.index,
                          Y=train_log_metric,
                          win=title,
                          opts=layout,
                          name='Training')
            if not val_log.empty:
                self.vis.line(X=val_log_metric.index,
                              Y=val_log_metric,
                              win=title,
                              opts=layout,
                              update='append',
                              name='Validation')
            return


except ImportError:
    VISDOM_AVAILABLE = False
    visdom = ModuleType('Unreachable module')

try:
    import plotly.express as px  # type: ignore

    PLOTLY_AVAILABLE: bool = True


    class PlotlyPlotter(BackendPlotter):
        def _plot(self,
                  train_log: DataFrame,
                  val_log: DataFrame,
                  loss_or_metric: str,
                  start: int,
                  title: str) -> None:
            train_log = train_log.copy()
            train_log['Dataset'] = "Training"
            val_log = val_log.copy()
            val_log['Dataset'] = "Validation"
            # noinspection PyUnreachableCode
            log = pd.concat((train_log, val_log))
            log = log[log.index >= start].reset_index().rename(
                columns={'index': 'Epoch'})
            fig = px.line(log,
                          x="Epoch",
                          y=loss_or_metric,
                          color="Dataset",
                          title=title)
            fig.show()
            return


except ImportError:
    PLOTLY_AVAILABLE = False
    px = ModuleType('Unreachable module')


class GetPlotterProtocol(Protocol):
    """
    This protocol is a Callable with named parameters.
    """

    def __call__(self, backend: Backend, env: str) -> BackendPlotter:
        ...


def plotter_closure() -> GetPlotterProtocol:
    """
    Cache the last used plotter.

    Return:
        A function that returns a new plotter backend or the last used plotter.
    """

    plotter: BackendPlotter = NoPlotter('')

    def get_plotter(backend: Backend = 'auto', env: str = '') -> BackendPlotter:
        """
        Returns a plotter object based on the specified backend.

        Args:
            backend:
                auto (default):
                    from a Jupiter notebook: use plotly and return ImportError
                    if plotly is not installed.
                    otherwise: if visdom is installed use visdom else plotly.
                     If neither are installed return NoPlotter.
                plotly: plotly backend. Return ImportError if
                plotly is not installed.
                visdom: visdom backend. Return ImportError if
                visdom is not installed.
            env: The optional environment model_name for the backend. Default ''.

        Return:
            A plotter backend based on the specified backend and environment.

        Raise:
            ImportError if the specified backend is not available.
            warning if the visdom connection is refused by the server.
        """
        nonlocal plotter
        jupyter = os.path.basename(os.environ['_']) == 'jupyter'
        # True when calling from a jupyter notebook
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
                raise exceptions.LibraryNotAvailableError('plotly')

            if not isinstance(plotter, PlotlyPlotter):
                plotter = PlotlyPlotter()

            return plotter

        if backend == 'visdom':
            if not VISDOM_AVAILABLE:
                raise exceptions.LibraryNotAvailableError('visdom')

            if isinstance(plotter, VisdomPlotter):
                if plotter.check_connection():
                    return plotter

            try:
                plotter = VisdomPlotter(env)
                return plotter
            except ConnectionError:
                warnings.warn(exceptions.VisdomConnectionWarning())
                return get_plotter('none', '')

        if backend == 'none':
            if not isinstance(plotter, NoPlotter):
                plotter = NoPlotter()

            return plotter

        raise exceptions.LibraryNotSupportedError(backend)

    return get_plotter


class Plotter:
    """
    Keep track of metrics and provide a plotting interface.

    Args:
         model_name: The model_name of the experiment.

    Methods:
        allow_extract_metadata: attempt to fully document the experiment.
        plot_learning_curves: plot the learning curves using an available
        backend.
    """

    def __init__(self, model_name: str = 'module') -> None:
        self.model_name = model_name
        self.get_plotter = plotter_closure()

    def plot_learning_curves(self,
                             metric: str = 'Criterion',
                             start: int = 0,
                             title: str = 'Learning Curves',
                             lib: Backend = 'auto') -> None:
        """
        This method plots the learning curves using either plotly or visdom as
        backends

        Args:
            metric: the loss_fun or the metric to visualize.
            start: the epoch from where you want to display the curve.
            title: the model_name of the window (and title) of the plot in the visdom
            interface.
            lib: which library to use as backend. 'auto' default to visdom or
            plotly if from a jupyter notebook.
        """
        log = tracking.Experiment.current().model_dict[self.model_name].log
        plotter = self.get_plotter(backend=lib, env=self.model_name)
        plotter.plot(log[data_types.Split.TRAIN],
                     log[data_types.Split.VAL],
                     metric,
                     start,
                     title)
        return

    def __repr__(self):
        return self.model_name

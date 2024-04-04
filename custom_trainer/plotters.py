import os
import sys
from types import ModuleType
from abc import ABCMeta, abstractmethod

import pandas as pd


class Plotter(metaclass=ABCMeta):
    def __init__(self, env: str = ''):
        self.env = env

    @abstractmethod
    def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
             loss_or_metric: str = 'Criterion', start: int = 0,
             title: str = 'Learning Curves') -> None:
        ...


class NoPlotter(Plotter):

    def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
             loss_or_metric: str = 'Criterion', start: int = 0,
             title: str = 'Learning Curves') -> None:
        pass


try:
    import visdom


    class VisdomPlotter(Plotter):

        def __init__(self, env: str = ''):
            super().__init__(env)
            self.env = env
            self.vis: visdom.Visdom = visdom.Visdom(env=env, raise_exceptions=True)

        def check_visdom_connection(self) -> bool:
            """
            It initializes a visdom environment with the name of the experiment (if it does not exist already)

            Returns:
                True visdom is available, False otherwise

            It activates and check the connection to the visdom server https://github.com/fossasia/visdom

            Returns:
                True if there is an active connection, False otherwise
            """
            return self.vis.check_connection()

        def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
                 loss_or_metric: str = 'Criterion', start: int = 0,
                 title: str = 'Learning Curves') -> None:
            """
            This class method plots the learning curves using visdom as backend. You need first to initialize an
            environment within the class attribute vis (see check_visdom_connection)

            Args:
                train_log: pandas Series with the loss and metrics calculated during training on the training dataset
                val_log: pandas Dataframe with the loss and metrics calculated during training on the validation dataset
                loss_or_metric: the Series or the metric to visualize
                start: the epoch from where you want to display the curve
                title: the name of the window (and title) of the plot in the visdom interface
            """

            train_log_metric: pd.Series[float] = train_log[train_log.index > start][loss_or_metric]
            val_log_metric: pd.Series[float] = val_log[val_log.index > start][loss_or_metric]
            layout = dict(xlabel='Epoch', ylabel=loss_or_metric, title=title, update='replace', showlegend=True)
            if self.check_visdom_connection():
                self.vis.line(X=train_log_metric.index, Y=train_log_metric, win=title, opts=layout, name='Training')
            if not val_log.empty:
                self.vis.line(X=val_log_metric.index, Y=val_log, win=title, opts=layout, update='append',
                              name='Validation')
            return


    VISDOM_AVAILABLE: bool = True

except ImportError:
    print("Plotting on visdom is disabled")
    VISDOM_AVAILABLE = False
    visdom = ModuleType('Unreachable module')

try:
    import plotly.graph_objects as px  # type: ignore


    class PlotlyPlotter(Plotter):
        def plot(self, train_log: pd.DataFrame, val_log: pd.DataFrame,
                 loss_or_metric: str = 'Criterion', start: int = 0,
                 title: str = 'Learning Curves') -> None:
            """
            This static method plots the learning curves using plotly as backend.

            Args:
                train_log: pandas Dataframe with the loss and metrics calculated during training on the training dataset
                val_log: pandas Dataframe with the loss and metrics calculated during training on the validation dataset
                loss_or_metric: the loss or the metric to visualize
                start: the epoch from where you want to display the curve
                title: the name of the window (and title) of the plot in the visdom interface
            """
            train_log = train_log.copy()
            train_log['Dataset'] = "Training"
            val_log = val_log.copy()
            val_log['Dataset'] = "Validation"
            log = pd.concat([train_log, val_log])
            log = log[log.index >= start].reset_index().rename(columns={'index': 'Epoch'})
            fig = px.line(log, x="Epoch", y=loss_or_metric, color="Dataset", title=title)  # type: ignore
            fig.show()
            return


    PLOTLY_AVAILABLE: bool = True
except ImportError:
    print("Plotting on plotly is disabled")
    PLOTLY_AVAILABLE = False
    px = ModuleType('Unreachable module')


def get_plotter(backend: str = 'auto', env: str = '') -> Plotter:
    jupyter = os.path.basename(os.environ['_']) == 'jupyter'  # True when calling from a jupyter notebook
    if backend == 'plotly' or jupyter:
        if PLOTLY_AVAILABLE:
            return PlotlyPlotter(env)
        else:
            raise ImportError(f'Library plotly not installed.')
    elif backend == 'auto':
        if VISDOM_AVAILABLE:
            try:
                plotter: VisdomPlotter = VisdomPlotter(env)
            except ConnectionError:
                print('Visdom connection refused by server', file=sys.stderr)
            else:
                if plotter.check_visdom_connection():
                    return plotter
        if PLOTLY_AVAILABLE:
            return PlotlyPlotter('')
        else:
            return NoPlotter('')
    elif backend == 'visdom' and VISDOM_AVAILABLE:
        if VISDOM_AVAILABLE:
            plotter = VisdomPlotter(env)
            if plotter.check_visdom_connection():
                return plotter
            else:
                print('Visdom connection refused by server', file=sys.stderr)
                return NoPlotter('')
        else:
            raise ImportError(f'Library visdom not installed.')
    else:
        raise ValueError(f'Library {backend} not supported.')

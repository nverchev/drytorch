"""Tracker for visualizations on visdom."""

import functools
from typing import Optional, Iterable, TypedDict

import numpy as np
from typing_extensions import override

import numpy.typing as npt
import visdom

from dry_torch.trackers import base_classes
from dry_torch import exceptions
from dry_torch import log_events


class VisdomOpts(TypedDict, total=False):
    """
    Annotations for optional settings in visdom.

    See: https://github.com/fossasia/visdom/blob/master/py/visdom/__init__.py.
    """
    title: str
    xlabel: str
    ylabel: str
    zlabel: str

    # Layout & Size
    width: int
    height: int
    marginleft: int
    marginright: int
    margintop: int
    marginbottom: int

    # Line style
    fillarea: bool
    markers: bool
    markersymbol: str
    markersize: float
    markercolor: npt.NDArray
    markerborderwidth: float
    mode: str
    dash: npt.NDArray[np.str_]  # e.g., np.array(['solid', 'dash'])

    # Axis limits
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    # Colors & Style
    linecolor: npt.NDArray  # shape: (num_lines, 3) RGB
    colormap: str  # matplotlib-style colormap name (for some plots)

    # Legend
    legend: list[str] | tuple[str, ...]
    showlegend: bool

    # Fonts
    titlefont: dict  # e.g., {"size": 16, "color": "red"}
    tickfont: dict

    # Misc
    opacity: float
    name: str
    textlabels: list[str]  # used in bar, scatter, etc.


class VisdomPlotter(base_classes.BasePlotter[str]):
    """
    Tracker that uses visdom as backend.

    Attributes:
        server: address for the visdom server.
        port: the port for the server.
    """

    def __init__(
            self,
            server: str = 'http://localhost',
            port: int = 8097,
            opts: Optional[VisdomOpts] = None,
            model_names: Iterable[str] = (),
            metric_names: Iterable[str] = (),
            metric_loader: Optional[base_classes.MetricLoader] = None,
            start: int = 1,
    ) -> None:
        """
        Args:
            model_names: the names of the models to plot. Defaults to all.
            metric_names: the names of the metrics to plot. Defaults to all.
            metric_loader: a tracker that can load metrics from a previous run.
            start: if positive the epoch from which to start plotting.
                if negative the last number of epochs. Defaults to all.
        """
        super().__init__(model_names, metric_names, metric_loader, start)
        self.server = server
        self.port = port
        self.opts: VisdomOpts = opts or {}
        self._viz: Optional[visdom.Visdom] = None

    @property
    def viz(self) -> visdom.Visdom:
        """The active Visdom instance."""
        if self._viz is None:
            raise exceptions.AccessOutsideScopeError()
        return self._viz

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        env = event.exp_name
        try:
            self._viz = visdom.Visdom(server=self.server,
                                      port=self.port,
                                      env=env,
                                      raise_exceptions=True)
        except ConnectionError as cre:
            msg = 'server not available.'
            raise exceptions.TrackerException(self, msg) from cre
        else:
            self.viz.close(env=event.exp_name)  # close all the windows
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self._viz = None
        return super().notify(event)

    @override
    def _plot_metric(self,
                     model_name: str,
                     metric_name: str,
                     **sources: npt.NDArray[np.float64]) -> str:

        layout = VisdomOpts(xlabel='Epoch',
                            ylabel=metric_name,
                            title=model_name,
                            showlegend=True)
        scatter_opts = VisdomOpts(mode='markers', markersymbol='24')
        opts = self.opts | layout
        win = '_'.join((model_name, metric_name))
        for name, log in sources.items():
            self.viz.scatter(None, win=win, update='remove', name=name)
            if log.shape[0] > 1:
                self.viz.line(X=log[:, 0],
                              Y=log[:, 1],
                              opts=opts,
                              update='append',
                              win=win,
                              name=name,
                              )
            else:
                self.viz.scatter(X=log[:, 0],
                                 Y=log[:, 1],
                                 opts=opts | scatter_opts,
                                 update='append',
                                 win=win,
                                 name=name,
                                 )

        return win

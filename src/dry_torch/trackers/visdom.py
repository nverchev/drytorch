"""Tracker for visualizations on visdom."""

import functools
from typing import Optional, Iterable, TypedDict

from typing_extensions import override

import numpy as np
import visdom

from dry_torch.trackers import base_classes
from dry_torch import exceptions
from dry_torch import log_events


class VisdomOpts(TypedDict, total=False):
    """
    Annotations for optional settings in visdom.

    See: https://github.com/fossasia/visdom/blob/master/py/visdom/__init__.py.
    """
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
    markercolor: np.ndarray
    markerborderwidth: float
    dash: np.ndarray  # e.g., np.array(['solid', 'dash'])

    # Axis limits
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    # Colors & Style
    linecolor: np.ndarray  # shape: (num_lines, 3) RGB
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


class VisdomPlotter(base_classes.BasePlotter):
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
            start_epoch: int = 1,
    ) -> None:
        """
        Args:
            model_names: the names of the models to plot. Defaults to all.
            metric_names: the names of the metrics to plot. Defaults to all.
            metric_loader: a tracker that can load metrics from a previous run.
            start_epoch: epoch from which plot starts.
        """
        super().__init__(model_names, metric_names, metric_loader, start_epoch)
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
        self._viz = visdom.Visdom(server=self.server, port=self.port, env=env)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperiment) -> None:
        self._viz = None
        return super().notify(event)

    @override
    def _plot_metric(self,
                     model_name: str,
                     metric_name: str,
                     **sources: tuple[list[int], list[float]]) -> None:

        layout = dict(xlabel='Epoch',
                      ylabel=metric_name,
                      title=model_name,
                      showlegend=True)

        opts = self.opts | layout
        win = '_'.join((model_name, metric_name))
        for source, (epochs, values) in sources.items():
            self.viz.scatter(None, None,
                             win=win, update='remove', name=source)
            self.viz.line(X=np.array(epochs),
                          Y=np.array(values),
                          opts=opts,
                          update='append',
                          win=win,
                          name=source,
                          )

        return

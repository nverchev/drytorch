"""Module containing a tracker for visualizations on visdom."""

import functools

from collections.abc import Iterable
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import visdom

from typing_extensions import override

from drytorch.core import exceptions, log_events
from drytorch.trackers import base_classes


class VisdomOpts(TypedDict, total=False):
    """Annotations for optional settings in visdom.

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
    markercolor: npt.NDArray[np.str_]
    markerborderwidth: float
    mode: str
    dash: npt.NDArray[np.str_]  # e.g., np.array(['solid', 'dash'])

    # Axis limits
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    # Colors & Style
    linecolor: npt.NDArray[Any]  # shape: (num_lines, 3) RGB
    colormap: str  # matplotlib-style colormap name (for some plots)

    # Legend
    legend: list[str] | tuple[str, ...]
    showlegend: bool

    # Fonts
    titlefont: dict[str, Any]  # e.g., {"size": 16, "color": "red"}
    tickfont: dict[str, Any]

    # Misc
    opacity: float
    name: str
    textlabels: list[str]  # used in bar, scatter, etc.


class VisdomPlotter(base_classes.BasePlotter[str]):
    """Tracker that uses visdom as backend.

    Attributes:
        server: the address for the visdom server.
        port: the port for the server.
    """

    def __init__(
        self,
        server: str = 'http://localhost',
        port: int = 8097,
        opts: VisdomOpts | None = None,
        model_names: Iterable[str] = (),
        source_names: Iterable[str] = (),
        metric_names: Iterable[str] = (),
        metric_loader: base_classes.MetricLoader | None = None,
        start: int = 1,
    ) -> None:
        """Constructor.

        Args:
            server: the address of the server.
            port: the port for the connection:
            opts: plot options.
            model_names: the names of the models to plot. Defaults to all.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            metric_loader: a tracker that can load metrics from a previous run.
            start: if positive, the epoch from which to start plotting;
                if negative, the last number of epochs. Defaults to all.
        """
        super().__init__(
            model_names, source_names, metric_names, start, metric_loader
        )
        self.server = server
        self.port = port
        self.opts: VisdomOpts = opts or VisdomOpts()
        self._viz: visdom.Visdom | None = None
        return

    @property
    def viz(self) -> visdom.Visdom:
        """The active Visdom instance."""
        if self._viz is None:
            raise exceptions.AccessOutsideScopeError()

        return self._viz

    @override
    def clean_up(self) -> None:
        self._viz = None
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        env = event.exp_name + '_' + event.run_id
        try:
            self._viz = visdom.Visdom(
                server=self.server,
                port=self.port,
                env=env,
                raise_exceptions=True,
            )
        except ConnectionError as cre:
            msg = 'server not available.'
            raise exceptions.TrackerError(self, msg) from cre
        else:
            self.viz.close(env=env)  # close all the windows

        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StopExperimentEvent) -> None:
        self.clean_up()
        return super().notify(event)

    @override
    def _plot_metric(
        self,
        model_name: str,
        metric_name: str,
        **sourced_array: base_classes.NpArray,
    ) -> str:
        layout = VisdomOpts(
            xlabel='Epoch',
            ylabel=metric_name,
            title=model_name,
            showlegend=True,
        )
        scatter_opts = VisdomOpts(mode='markers', markersymbol='24')
        opts = self.opts | layout  # type: ignore
        win = '_'.join((model_name, metric_name))
        for name, log in sourced_array.items():
            self.viz.scatter(None, win=win, update='remove', name=name)
            if log.shape[0] > 1:
                self.viz.line(
                    X=log[:, 0],
                    Y=log[:, 1],
                    opts=opts,
                    update='append',
                    win=win,
                    name=name,
                )
            else:
                self.viz.scatter(
                    X=log[:, 0],
                    Y=log[:, 1],
                    opts=opts | scatter_opts,
                    update='append',
                    win=win,
                    name=name,
                )

        return win

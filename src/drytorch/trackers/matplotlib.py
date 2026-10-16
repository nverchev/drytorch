"""Plotting with matplotlib."""

import math
import pathlib

from collections.abc import Iterable
from typing import ClassVar, TypeAlias

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import axes, colors, figure, ticker
from typing_extensions import override

from drytorch.core import exceptions
from drytorch.trackers import base_classes


__all__ = [
    'MatPlotter',
]


ColorType: TypeAlias = tuple[float, float, float, float] | str


class MatPlotter(base_classes.BasePlotter[figure.Figure]):
    """Tracker that plots each metric of a model in a figure with matplotlib."""

    file_suffix: ClassVar[str] = '.png'

    _cmap: colors.Colormap
    _source_colors: dict[str, ColorType]

    def __init__(
        self,
        *,
        metric_loader: base_classes.MetricLoader | None = None,
        par_dir: pathlib.Path | None = None,
        palette: str = 'tab10',
    ) -> None:
        """Initialize.

        Args:
            metric_loader: a tracker that can load metrics from a previous run.
            par_dir: the parent directory for the saved figures. Default uses
                the same of the current experiment.
            palette: colormap name to use for source colors (default: 'tab10').

        Raises:
            ValueError: if palette is not a valid colormap name.
        """
        super().__init__(metric_loader=metric_loader, par_dir=par_dir)
        self._cmap = self._get_cmap(palette)
        self._source_colors = {}
        return

    @override
    def close(self) -> None:
        """Release tracker state."""
        self._source_colors.clear()
        return super().close()

    def plot_grid(
        self,
        model_name: str,
        *,
        metric_names: Iterable[str] = (),
        source_names: Iterable[str] = (),
        start_epoch: int = 1,
    ) -> None:
        """Render the metrics as subplots of one figure.

        Args:
            model_name: the name of the model to plot.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            start_epoch: the epoch from which to start plotting.

        Raises:
            ValueError: if start_epoch is less than 1.
            ValueError: if the model is not found.
            ValueError: if no metric is left to plot.
        """
        metric_arrays = self._get_metric_arrays(
            model_name, metric_names, source_names, start_epoch
        )
        n_metrics = len(metric_arrays)
        n_rows = math.ceil(math.sqrt(n_metrics))
        n_cols = math.ceil(n_metrics / n_rows)
        fig = plt.figure()
        fig.suptitle(model_name, fontsize=16)
        for index, (metric_name, sourced_array) in enumerate(
            metric_arrays.items()
        ):
            ax = fig.add_subplot(n_rows, n_cols, index + 1)
            self._draw_metric(ax, metric_name, sourced_array)

        fig.tight_layout()
        plt.show()
        return

    @override
    def _create_figure(
        self,
        model_name: str,
        metric_name: str,
        sourced_array: base_classes.SourcedArray,
    ) -> figure.Figure:
        # a figure without pyplot is not rendered and keeps no global state
        fig = figure.Figure()
        ax = fig.add_subplot()
        ax.set_title(model_name)
        self._draw_metric(ax, metric_name, sourced_array)
        fig.tight_layout()
        return fig

    def _draw_metric(
        self,
        ax: axes.Axes,
        metric_name: str,
        sourced_array: base_classes.SourcedArray,
    ) -> None:
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        for name, log in sourced_array.items():
            color = self._get_source_color(name)
            if log.shape[0] == 1:
                ax.plot(
                    log[:, 0],
                    log[:, 1],
                    marker='D',
                    markersize=10,
                    linestyle='None',
                    color=color,
                    label=name,
                )
            else:
                ax.plot(log[:, 0], log[:, 1], color=color, label=name)

        ax.legend()
        return

    def _get_cmap(self, palette: str) -> colors.Colormap:
        try:
            return matplotlib.colormaps[palette]
        except (KeyError, TypeError) as exc:
            msg = f'Colormap {palette!r} is not registered in matplotlib.'
            raise ValueError(msg) from exc

    def _get_source_color(self, name: str) -> ColorType:
        if name not in self._source_colors:
            idx = len(self._source_colors)
            if idx >= self._cmap.N:
                msg = (
                    f'Palette {self._cmap.name!r} is insufficient: '
                    f'provides only {self._cmap.N} colors.'
                )
                raise exceptions.TrackerError(self, msg)

            if self._cmap.N <= 20:  # discrete colormap
                self._source_colors[name] = self._cmap(idx)
            else:
                self._source_colors[name] = self._cmap(
                    self.bisection_ratio(idx)
                )

        return self._source_colors[name]

    @override
    def _render(self, figures: list[figure.Figure]) -> None:
        for fig in figures:
            plt.figure(fig)  # pyplot manages the figure to show it

        plt.show()
        return

    @override
    def _save_figure(self, figure: figure.Figure, path: pathlib.Path) -> None:
        figure.savefig(path)
        return

    @staticmethod
    def bisection_ratio(idx: int) -> float:
        """Compute a ratio in [0, 1) by successive interval bisection."""
        ratio, step = 0.0, 0.5
        while idx > 0:
            if idx % 2:
                ratio += step

            idx //= 2
            step /= 2

        return ratio

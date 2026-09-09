"""Plotting with matplotlib."""

import math

from collections.abc import Iterable
from typing import TypeAlias

import matplotlib.pyplot as plt

from matplotlib import axes, colormaps, colors, figure
from typing_extensions import override

from drytorch.core import exceptions
from drytorch.trackers import base_classes


__all__ = [
    'MatPlotter',
]


ColorType: TypeAlias = tuple[float, float, float, float] | str
Plot: TypeAlias = tuple[figure.Figure, axes.Axes]


class MatPlotter(base_classes.BasePlotter[Plot]):
    """Tracker that organizes metrics as subplots using matplotlib."""

    _cmap: colors.Colormap
    _model_figure: dict[str, tuple[figure.Figure, dict[str, axes.Axes]]]
    _source_colors: dict[str, ColorType]

    def __init__(
        self,
        model_names: Iterable[str] = (),
        source_names: Iterable[str] = (),
        metric_names: Iterable[str] = (),
        metric_loader: base_classes.MetricLoader | None = None,
        start: int = 1,
        palette: str = 'tab10',
    ) -> None:
        """Initialize.

        Args:
            model_names: the names of the models to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            metric_names: the names of the metrics to plot. Defaults to all.
            metric_loader: a tracker that can load metrics from a previous run.
            start: if positive, the epoch from which to start plotting;
                if negative, the last number of epochs. Defaults to all.
            palette: colormap name to use for source colors (default: 'tab10').

        Raises:
            ValueError: if palette is not a valid colormap name.
        """
        super().__init__(
            model_names, source_names, metric_names, start, metric_loader
        )
        self._model_figure = {}
        self._cmap = self._get_cmap(palette)
        self._source_colors = {}
        plt.ion()
        return

    @override
    def close(self) -> None:
        """Release tracker state and close all figures."""
        for fig, _ in self._model_figure.values():
            plt.close(fig)

        self._model_figure.clear()
        self._source_colors.clear()
        return super().close()

    @override
    def _display_plot(self, model_name: str, plots: list[Plot]) -> None:
        _not_used = plots
        if model_name in self._model_figure:
            fig, _ = self._model_figure[model_name]
            fig.canvas.draw()
            fig.canvas.flush_events()

        return

    def _get_cmap(self, palette: str) -> colors.Colormap:
        try:
            return colormaps[palette]
        except (KeyError, TypeError) as exc:
            msg = f'Colormap {palette!r} is not registered in matplotlib.'
            raise ValueError(msg) from exc

    def _prepare_layout(self, model_name: str, metric_names: list[str]) -> None:
        if model_name in self._model_figure:
            fig, axes_dict = self._model_figure[model_name]
            if set(metric_names).issubset(axes_dict):
                return

            fig.clear()
            metric_names = sorted(set(axes_dict) | set(metric_names))
        else:
            fig = plt.figure()
            plt.show(block=False)

        n_metrics = len(metric_names)
        if n_metrics == 0:
            return

        n_rows = math.ceil(math.sqrt(n_metrics))
        n_cols = math.ceil(n_metrics / n_rows)
        fig.suptitle(model_name, fontsize=16)
        axes_dict = dict[str, axes.Axes]()
        for index, metric_name in enumerate(metric_names):
            ax = fig.add_subplot(n_rows, n_cols, index + 1)
            ax.set_title(metric_name)
            axes_dict[metric_name] = ax

        fig.tight_layout()
        self._model_figure[model_name] = (fig, axes_dict)
        return

    @override
    def _plot_metric(
        self,
        model_name: str,
        metric_name: str,
        **sourced_array: base_classes.NpArray,
    ) -> Plot:
        fig, dict_axes = self._model_figure[model_name]
        if metric_name not in dict_axes:
            self._prepare_layout(model_name, [metric_name])
            fig, dict_axes = self._model_figure[model_name]

        ax = dict_axes[metric_name]
        for collection in ax.collections[:]:
            collection.remove()

        dict_lines = {line.get_label(): line for line in ax.get_lines()}
        for name, log in sourced_array.items():
            if name in dict_lines:
                line = dict_lines[name]
                line.set_xdata(log[:, 0])
                line.set_ydata(log[:, 1])
                if log.shape[0] > 1 and line.get_marker() != 'None':
                    line.set_marker('None')
                    line.set_linestyle('-')

            elif log.shape[0] == 1:
                ax.plot(
                    log[:, 0],
                    log[:, 1],
                    marker='D',
                    markersize=10,
                    linestyle='None',
                    color=self._get_source_color(name),
                    label=name,
                )
            else:
                ax.plot(
                    log[:, 0],
                    log[:, 1],
                    color=self._get_source_color(name),
                    label=name,
                )

        ax.relim()
        ax.autoscale_view()
        ax.legend()
        return fig, ax

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

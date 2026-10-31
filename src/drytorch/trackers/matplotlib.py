"""Plotting with matplotlib."""

import pathlib

from collections.abc import Sequence
from typing import ClassVar, TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import artist, axes, colors, figure, ticker
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
    folder_name: ClassVar[str] = 'matplotlib'

    _cmap: colors.Colormap
    _source_colors: dict[str, ColorType]

    def __init__(
        self,
        *,
        metric_loader: base_classes.MetricLoader | None = None,
        par_dir: pathlib.Path | None = None,
        palette: str = 'tab10',
        show: bool = False,
    ) -> None:
        """Initialize.

        Args:
            metric_loader: a tracker that can load metrics from a previous run.
            par_dir: the parent directory for the saved figures. Default uses
                the same of the current experiment.
            palette: colormap name to use for source colors (default: 'tab10').
            show: whether to render the plots when saved.

        Raises:
            ValueError: if palette is not a valid colormap name.
        """
        super().__init__(
            metric_loader=metric_loader, par_dir=par_dir, show=show
        )
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
        metric_names: Sequence[str] = (),
        source_names: Sequence[str] = (),
        start_epoch: int = 1,
        exp_names: Sequence[str] = (),
    ) -> None:
        """Render the metrics as subplots of one figure, and the tests apart.

        Args:
            model_name: the name of the model to plot.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            start_epoch: the epoch from which to start plotting.
            exp_names: the names of the experiments to plot. Defaults to all.

        Raises:
            AccessOutsideScopeError: when no run is active.
            NothingToPlotError: if nothing is left to plot.
            ValueError: if start_epoch is less than 1.
        """
        metric_arrays = self._get_metric_arrays(
            model_name, metric_names, source_names, start_epoch
        )
        test_results = self._get_test_results(
            model_name,
            metric_names=metric_names,
            source_names=source_names,
            exp_names=exp_names,
        )
        if test_results:
            test_plot = self._create_test_plot(model_name, test_results)
        else:
            test_plot = None

        if not metric_arrays and not test_results:
            raise exceptions.NothingToPlotError(self, model_name)

        if metric_arrays:
            n_rows, n_cols = self._get_grid_shape(len(metric_arrays))
            fig = plt.figure()
            fig.suptitle(model_name, fontsize=16)
            for index, (metric_name, sourced_array) in enumerate(
                metric_arrays.items()
            ):
                ax = fig.add_subplot(n_rows, n_cols, index + 1)
                self._draw_metric(ax, metric_name, sourced_array)

            fig.tight_layout()

        if test_plot is not None:
            plt.figure(test_plot)  # pyplot manages the figure to show it

        plt.show()
        return

    @override
    def _create_training_plot(
        self,
        model_name: str,
        metric_name: str,
        sourced_array: base_classes.SourcedArray,
    ) -> figure.Figure:
        fig = figure.Figure()
        ax = fig.add_subplot()
        ax.set_title(model_name)
        self._draw_metric(ax, metric_name, sourced_array)
        fig.tight_layout()
        return fig

    @override
    def _create_test_plot(
        self, model_name: str, results: base_classes.ExperimentMetrics
    ) -> figure.Figure:
        metric_names = self._get_test_metric_names(results)
        n_rows, n_cols = self._get_grid_shape(len(metric_names))
        fig = figure.Figure(layout='constrained')
        fig.suptitle(f'{model_name} tests')
        for index, metric_name in enumerate(metric_names):
            ax = fig.add_subplot(n_rows, n_cols, index + 1)
            self._draw_test_metric(ax, metric_name, results)

        legend: dict[str, artist.Artist] = {}
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            for label, handle in zip(labels, handles, strict=True):
                if label not in legend:
                    legend[label] = handle

        fig.legend(legend.values(), legend.keys(), loc='outside lower center')
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

    def _draw_test_metric(
        self,
        ax: axes.Axes,
        metric_name: str,
        results: base_classes.ExperimentMetrics,
    ) -> None:
        test_names = self._get_test_names(results, metric_name)
        x_indices = np.arange(len(test_names))
        exp_list = list(results.keys())
        n_exp = len(exp_list)
        total_width = 0.8
        bar_width = total_width / n_exp
        for i, exp_name in enumerate(exp_list):
            color = self._get_source_color(exp_name)
            offset = (i - (n_exp - 1) / 2) * bar_width
            labeled = False
            for j, test_name in enumerate(test_names):
                value = self._get_last_value(
                    results[exp_name], test_name, metric_name
                )
                if value is not None:
                    label = exp_name if not labeled else None
                    ax.bar(
                        x_indices[j] + offset,
                        value,
                        width=bar_width,
                        color=color,
                        label=label,
                    )
                    labeled = True

        ax.set_xticks(x_indices, test_names)
        ax.set_title(metric_name)
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
    def _save_plot(self, figure: figure.Figure, path: pathlib.Path) -> None:
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

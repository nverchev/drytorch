"""Plotting with matplotlib."""

import pathlib

from collections.abc import Sequence
from typing import ClassVar, TypeAlias

import matplotlib
import matplotlib.pyplot as plt

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
    ) -> None:
        """Render the metrics as subplots of one figure, and the tests apart.

        Args:
            model_name: the name of the model to plot.
            metric_names: the names of the metrics to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            start_epoch: the epoch from which to start plotting.

        Raises:
            ValueError: if start_epoch is less than 1.
            ValueError: if neither a metric nor a test is left to plot.
        """
        metric_arrays = self._get_metric_arrays(
            model_name, metric_names, source_names, start_epoch
        )
        test_plot = self.create_test_plot(
            model_name, metric_names=metric_names, source_names=source_names
        )
        if not metric_arrays and test_plot is None:
            msg = f'No metric to plot for model {model_name}.'
            raise ValueError(msg)

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
        self, model_name: str, test_results: base_classes.TestResults
    ) -> figure.Figure:
        metric_names = self._get_test_metric_names(test_results)
        n_rows, n_cols = self._get_grid_shape(len(metric_names))
        fig = figure.Figure(layout='constrained')
        fig.suptitle(f'{model_name} tests')
        for index, metric_name in enumerate(metric_names):
            ax = fig.add_subplot(n_rows, n_cols, index + 1)
            self._draw_test_metric(ax, metric_name, test_results)

        legend: dict[str, artist.Artist] = {}
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            legend |= dict(zip(labels, handles, strict=True))

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
        test_results: base_classes.TestResults,
    ) -> None:
        positions: list[int] = []
        tick_labels: list[str] = []
        position = 0
        for source_name, entries in test_results.items():
            labeled = False
            for epoch, metrics in entries:
                if metric_name in metrics:
                    label = source_name if not labeled else None
                    ax.bar(
                        position,
                        metrics[metric_name],
                        color=self._get_source_color(source_name),
                        label=label,
                    )
                    positions.append(position)
                    tick_labels.append(str(epoch))
                    position += 1
                    labeled = True

        ax.set_xticks(positions, tick_labels)
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

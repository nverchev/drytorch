"""Plotting with matplotlib."""

import math

from collections.abc import Iterable
from typing import Final, TypeAlias

import matplotlib.pyplot as plt

from matplotlib import axes, figure
from typing_extensions import override

from drytorch.trackers import base_classes


Plot: TypeAlias = tuple[figure.Figure, axes.Axes]


class MatPlotter(base_classes.BasePlotter[Plot]):
    """Tracker that organizes metrics as subplots using matplotlib."""

    def __init__(
        self,
        model_names: Iterable[str] = (),
        source_names: Iterable[str] = (),
        metric_names: Iterable[str] = (),
        metric_loader: base_classes.MetricLoader | None = None,
        start: int = 1,
    ) -> None:
        """Constructor.

        Args:
            model_names: the names of the models to plot. Defaults to all.
            source_names: the names of the sources to plot. Defaults to all.
            metric_names: the names of the metrics to plot. Defaults to all.
            metric_loader: a tracker that can load metrics from a previous run.
            start: if positive, the epoch from which to start plotting;
                if negative, the last number of epochs. Defaults to all.
        """
        super().__init__(
            model_names, source_names, metric_names, start, metric_loader
        )
        self._model_figure: Final = dict[
            str, tuple[figure.Figure, dict[str, axes.Axes]]
        ]()
        plt.ion()
        return

    def _prepare_layout(self, model_name: str, metric_names: list[str]) -> None:
        if model_name not in self._model_figure:
            n_metrics = len(metric_names)
            n_rows = math.ceil(math.sqrt(n_metrics))
            n_cols = math.ceil(n_metrics / n_rows)
            fig = figure.Figure()
            fig.suptitle(model_name, fontsize=16)
            fig.tight_layout()
            iter_metric = iter(metric_names)
            axes_dict = dict[str, axes.Axes]()
            for index in range(n_metrics):
                try:
                    metric_name = next(iter_metric)
                except StopIteration:
                    break
                else:
                    ax = fig.add_subplot(n_rows, n_cols, index + 1)
                    axes_dict[metric_name] = ax

            self._model_figure[model_name] = (fig, axes_dict)
            plt.show(block=False)

    @override
    def _plot_metric(
        self,
        model_name: str,
        metric_name: str,
        **sourced_array: base_classes.NpArray,
    ) -> Plot:
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
            elif log.shape[0] == 1:
                # Create a scatter plot for a single point
                ax.scatter(log[:, 0], log[:, 1], s=200, label=name, marker='D')
            else:
                ax.plot(log[:, 0], log[:, 1], label=name)

        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, ax

    def close_all(self):
        """Close all figures associated with this plotter."""
        for fig, _ in self._model_figure.values():
            plt.close(fig)

        self._model_figure.clear()

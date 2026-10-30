"""Plotter class using plotly."""

import pathlib

from typing import ClassVar

import plotly.graph_objs as go

from plotly import colors as pc
from plotly import subplots
from typing_extensions import override

from drytorch.trackers import base_classes


__all__ = [
    'PlotlyPlotter',
]


class PlotlyPlotter(base_classes.BasePlotter[go.Figure]):
    """Tracker that plots each metric of a model in a figure with plotly."""

    file_suffix: ClassVar[str] = '.html'
    folder_name: ClassVar[str] = 'plotly'

    @override
    def _create_training_plot(
        self,
        model_name: str,
        metric_name: str,
        sourced_array: base_classes.SourcedArray,
    ) -> go.Figure:
        data = [
            go.Scatter(x=log[:, 0], y=log[:, 1], name=name)
            for name, log in sourced_array.items()
        ]
        return go.Figure(
            data=data,
            layout=go.Layout(
                title=model_name,
                xaxis={'title': 'Epoch'},
                yaxis={'title': metric_name},
            ),
        )

    @override
    def _create_test_plot(
        self, model_name: str, test_results: base_classes.TestResults
    ) -> go.Figure:
        metric_names = self._get_test_metric_names(test_results)
        n_rows, n_cols = self._get_grid_shape(len(metric_names))
        figure = subplots.make_subplots(
            rows=n_rows, cols=n_cols, subplot_titles=metric_names
        )
        palette = pc.DEFAULT_PLOTLY_COLORS
        colors = {
            source: palette[i % len(palette)]
            for i, source in enumerate(test_results)
        }
        seen_sources: set[str] = set()
        for index, metric_name in enumerate(metric_names):
            row, col = divmod(index, n_cols)
            position = 0
            positions: list[int] = []
            tick_labels: list[str] = []
            for source_name, entries in test_results.items():
                x_vals: list[int] = []
                y_vals: list[float] = []
                for epoch, metrics in entries:
                    if metric_name in metrics:
                        x_vals.append(position)
                        y_vals.append(metrics[metric_name])
                        positions.append(position)
                        tick_labels.append(str(epoch))
                        position += 1

                if x_vals:
                    show_legend = source_name not in seen_sources
                    seen_sources.add(source_name)
                    bar = go.Bar(
                        x=x_vals,
                        y=y_vals,
                        name=source_name,
                        legendgroup=source_name,
                        showlegend=show_legend,
                        marker_color=colors[source_name],
                    )
                    figure.add_trace(bar, row=row + 1, col=col + 1)

            figure.update_xaxes(
                tickmode='array',
                tickvals=positions,
                ticktext=tick_labels,
                row=row + 1,
                col=col + 1,
            )

        figure.update_layout(title=f'{model_name} tests')
        return figure

    @override
    def _render(self, figures: list[go.Figure]) -> None:
        for figure in figures:
            figure.show()

        return

    @override
    def _save_plot(self, figure: go.Figure, path: pathlib.Path) -> None:
        figure.write_html(path)
        return

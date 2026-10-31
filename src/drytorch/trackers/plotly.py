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
        self, model_name: str, results: base_classes.ExperimentMetrics
    ) -> go.Figure:
        metric_names = self._get_test_metric_names(results)
        n_rows, n_cols = self._get_grid_shape(len(metric_names))
        figure = subplots.make_subplots(
            rows=n_rows, cols=n_cols, subplot_titles=metric_names
        )
        palette = pc.DEFAULT_PLOTLY_COLORS
        colors = {
            exp_name: palette[i % len(palette)]
            for i, exp_name in enumerate(results)
        }
        seen_experiments: set[str] = set()
        for index, metric_name in enumerate(metric_names):
            row, col = divmod(index, n_cols)
            test_names = self._get_test_names(results, metric_name)
            for exp_name, sourced_metrics in results.items():
                y_vals: list[float] = []
                for test_name in test_names:
                    value = self._get_last_value(
                        sourced_metrics, test_name, metric_name
                    )
                    y_vals.append(float('nan') if value is None else value)

                show_legend = exp_name not in seen_experiments
                seen_experiments.add(exp_name)
                bar = go.Bar(
                    x=test_names,
                    y=y_vals,
                    name=exp_name,
                    legendgroup=exp_name,
                    showlegend=show_legend,
                    marker_color=colors[exp_name],
                )
                figure.add_trace(bar, row=row + 1, col=col + 1)

        figure.update_layout(title=f'{model_name} tests', barmode='group')
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

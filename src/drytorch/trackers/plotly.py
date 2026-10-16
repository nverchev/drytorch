"""Plotter class using plotly."""

import pathlib

from typing import ClassVar

import plotly.graph_objs as go

from typing_extensions import override

from drytorch.trackers import base_classes


__all__ = [
    'PlotlyPlotter',
]


class PlotlyPlotter(base_classes.BasePlotter[go.Figure]):
    """Tracker that plots each metric of a model in a figure with plotly."""

    file_suffix: ClassVar[str] = '.html'

    @override
    def _create_figure(
        self,
        model_name: str,
        metric_name: str,
        sourced_array: base_classes.SourcedArray,
    ) -> go.Figure:
        data: list[go.Scatter] = []
        for name, log in sourced_array.items():
            if log.shape[0] == 1:
                marker = go.scatter.Marker(symbol='diamond', size=20)
                data.append(
                    go.Scatter(
                        x=log[:, 0],
                        y=log[:, 1],
                        mode='markers',
                        marker=marker,
                        name=name,
                    )
                )
            else:
                data.append(go.Scatter(x=log[:, 0], y=log[:, 1], name=name))

        return go.Figure(
            data=data,
            layout=go.Layout(
                title=model_name,
                xaxis={'title': 'Epoch'},
                yaxis={'title': metric_name},
            ),
        )

    @override
    def _render(self, figures: list[go.Figure]) -> None:
        for figure in figures:
            figure.show()

        return

    @override
    def _save_figure(self, figure: go.Figure, path: pathlib.Path) -> None:
        figure.write_html(path)
        return

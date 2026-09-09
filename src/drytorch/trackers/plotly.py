"""Module containing a plotter class using plotly."""

import plotly.graph_objs as go

from typing_extensions import override

from drytorch.trackers import base_classes


__all__ = [
    'PlotlyPlotter',
]


class PlotlyPlotter(base_classes.BasePlotter[go.Figure]):
    """Tracker that builds a plotly figure per metric on request."""

    @override
    def _display_plot(self, model_name: str, plots: list[go.Figure]) -> None:
        """Render the figures once a requested plot pass is complete."""
        _not_used = model_name
        for figure in plots:
            figure.show()

        return

    @override
    def _plot_metric(
        self,
        model_name: str,
        metric_name: str,
        **sourced_array: base_classes.NpArray,
    ) -> go.Figure:
        data = list[go.Scatter]()
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
    def _update_plot(self, model_name: str, start: int) -> None:
        """Skip the automatic per-epoch pass."""
        _not_used = model_name, start
        return

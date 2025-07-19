"""Module containing a plotter class using plotly."""

import plotly.graph_objs as go

from typing_extensions import override

from drytorch.trackers import base_classes


class PlotlyPlotter(base_classes.BasePlotter[go.Figure]):
    """Tracker that creates new plots each call (no update) using plotly."""

    @override
    def _plot_metric(
        self,
        model_name: str,
        metric_name: str,
        **sourced_array: base_classes.NpArray,
    ) -> go.Figure:
        data = list[go.Scatter | go.Bar]()
        for name, log in sourced_array.items():
            if log.shape[0] == 1:
                marker = go.scatter.Marker(symbol=24, size=20)
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

        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title=model_name,
                xaxis={'title': 'Epoch'},
                yaxis={'title': metric_name},
            ),
        )
        fig.show()
        return fig

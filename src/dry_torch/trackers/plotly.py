"""Plotting with plotly."""
from typing_extensions import override

from dry_torch.trackers.base_classes import BasePlotter

import numpy as np
import numpy.typing as npt
import plotly.graph_objs as go  # type: ignore


class PlotlyPlotter(BasePlotter[go.Figure]):
    """Tracker that create new plots each call (no update) using plotly."""

    @override
    def _plot_metric(self,
                     model_name: str,
                     metric_name: str,
                     **sources: npt.NDArray[np.float64]) -> go.Figure:
        data = list[go.Scatter | go.Bar]()

        for name, log in sources.items():
            if log.shape[0] == 1:
                marker = go.scatter.Marker(symbol=24, size=20)
                data.append(go.Scatter(x=log[:, 0],
                                       y=log[:, 1],
                                       mode='markers',
                                       marker=marker,
                                       name=name))
            else:
                data.append(go.Scatter(x=log[:, 0], y=log[:, 1], name=name))

        fig = go.Figure(data=data,
                        layout=go.Layout(
                            title=model_name,
                            xaxis=dict(title='Epoch'),
                            yaxis=dict(title=metric_name)
                            )
                        )
        fig.show()
        return fig

"""Plotting with plotly."""

from dry_torch.trackers.base_classes import BasePlotter

import plotly.graph_objs as go  # type: ignore


class PlotlyPlotter(BasePlotter):

    def _plot_metric(self,
                     model_name: str,
                     metric_name: str,
                     **sources: tuple[list[int], list[float]]) -> None:
        data = list[go.Scatter]()
        for name, source in sources.items():
            data.append(go.Scatter(x=source[0], y=source[1], name=name))
        fig = go.Figure(data=data,
                        layout=go.Layout(
                            title=model_name,
                            xaxis=dict(title='Epoch'),
                            yaxis=dict(title=metric_name)
                            )
                        )
        fig.show()
        return

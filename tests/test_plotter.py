from custom_trainer.plotters import Plotter, VisdomPlotter, PlotlyPlotter, GetPlotterProtocol, plotter_backend

import pandas as pd

DEFAULT_DATAFRAME = pd.DataFrame({'Criterion': [0, 1]})


def test_PlotlyPlotter():
    plotter = PlotlyPlotter()
    plotter.plot(DEFAULT_DATAFRAME, DEFAULT_DATAFRAME)


def test_VisdomPlotter():
    plotter = VisdomPlotter('test')
    plotter.plot(DEFAULT_DATAFRAME, DEFAULT_DATAFRAME)


def test_plotter_backend():
    get_plotter = plotter_backend()

    # this tests depends on the system settings
    plotter = get_plotter(backend='auto', env='')
    print(type(plotter))

    # test caching
    assert plotter is get_plotter(backend='auto', env='test')

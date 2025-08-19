"""Tests for the "base_classes" module."""

import functools
import pathlib

from collections.abc import Generator

import numpy as np

from typing_extensions import override

import pytest

from drytorch.core import exceptions, log_events
from drytorch.trackers.base_classes import (
    BasePlotter,
    Dumper,
    MemoryMetrics,
    MetricLoader,
    SourcedMetrics,
)


@pytest.fixture(scope='module')
def example_sourced_metrics(
    example_source_name, example_loss_name
) -> SourcedMetrics:
    """Example of sourced metrics."""
    return {
        'short_source': (
            [1, 2],
            {example_loss_name: [3.0, 4.0], 'example_metric_2': [6.0, 6.0]},
        ),
        example_source_name: (
            [1, 2, 4],
            {
                example_loss_name: [3.0, 4.0, 5],
                'example_metric': [6.0, 6.0, 7.0],
            },
        ),
        'source_added_later': (
            [1, 2, 4],
            {
                example_loss_name: [3.0, 4.0, 5],
                'example_metric': [6.0, 6.0, 7.0],
            },
        ),
        'empty_source': ([], {}),
    }


class _ConcreteMetricLoader(MetricLoader):
    def __init__(self, sourced_metrics: SourcedMetrics):
        super().__init__()
        self.sourced_metrics = sourced_metrics

    def _load_metrics(
        self, model_name: str, max_epoch: int = -1
    ) -> SourcedMetrics:
        if model_name == 'wrong_name':
            return {}
        return self.sourced_metrics

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return


class _ConcretePlotter(BasePlotter[str]):
    def _plot_metric(self, model_name, metric_name, **sources) -> str:
        return model_name + metric_name


@pytest.fixture(scope='module')
def metric_loader(example_sourced_metrics) -> MetricLoader:
    """Set up the instance."""
    return _ConcreteMetricLoader(example_sourced_metrics)


class TestDumper:
    """Tests for the Dumper class."""

    @pytest.fixture(
        scope='class', params=[None, 'test'], ids=['default', 'custom_path']
    )
    def par_dir(self, request, tmp_path_factory) -> pathlib.Path | None:
        """Set up the path."""
        return None if request.param is None else tmp_path_factory.mktemp('tmp')

    @pytest.fixture
    def tracker(self, par_dir) -> Dumper:
        """Set up the instance."""
        return Dumper(par_dir=par_dir)

    @pytest.fixture
    def tracker_started(
        self, tracker, start_experiment_mock_event, stop_experiment_mock_event
    ) -> Generator[Dumper, None, None]:
        """Start the instance."""
        tracker.notify(start_experiment_mock_event)
        yield tracker
        tracker.notify(stop_experiment_mock_event)
        return

    def test_par_dir_fails(self, par_dir, tracker) -> None:
        """Test par_dir property fails when outside scope."""
        if par_dir is None:
            with pytest.raises(exceptions.AccessOutsideScopeError):
                _ = tracker.par_dir

    def test_par_dir_success(
        self, par_dir, tracker_started, start_experiment_mock_event
    ) -> None:
        """Test par_dir property inside scope."""
        exp_par_dir = start_experiment_mock_event.par_dir
        if par_dir is None:
            assert tracker_started.par_dir == exp_par_dir
        else:
            assert tracker_started.par_dir == par_dir


class TestMetricLoader:
    """Tests for Metric Loader."""

    @pytest.fixture
    def loader_started(
        self,
        metric_loader,
        start_experiment_mock_event,
        stop_experiment_mock_event,
    ) -> Generator[MetricLoader, None, None]:
        """Start the instance."""
        metric_loader.notify(start_experiment_mock_event)
        yield metric_loader
        metric_loader.notify(stop_experiment_mock_event)
        return

    def test_correct_functioning(
        self, example_model_name, loader_started, example_sourced_metrics
    ) -> None:
        """Test correct functioning."""
        loaded_metrics = loader_started.load_metrics(example_model_name)
        assert loaded_metrics == example_sourced_metrics

    def test_null(self, example_model_name, loader_started) -> None:
        """Test empty dictionary return."""
        assert loader_started.load_metrics(example_model_name, 0) == {}

    def test_value_error(self, example_model_name, loader_started) -> None:
        """Test value error."""
        with pytest.raises(ValueError):
            _ = loader_started.load_metrics(example_model_name, -2)


class TestMemoryMetrics:
    """Tests for MemoryMetrics."""

    @pytest.fixture(scope='class')
    def tracker(self, metric_loader) -> MemoryMetrics:
        """Set up the instance."""
        return MemoryMetrics(metric_loader)

    def test_initialization(self, tracker) -> None:
        """Test basic initialization."""
        assert tracker.model_dict == {}

    def test_metrics_notification(
        self, example_model_name, tracker, epoch_metrics_mock_event
    ) -> None:
        """Test notification with Metrics event."""
        tracker.notify(epoch_metrics_mock_event)
        example_model_name = epoch_metrics_mock_event.model_name
        source_name = epoch_metrics_mock_event.source_name
        sample_metrics = epoch_metrics_mock_event.metrics
        assert example_model_name in tracker.model_dict
        assert source_name in tracker.model_dict[example_model_name]
        epochs, logs = tracker.model_dict[example_model_name][source_name]
        assert epochs == [epoch_metrics_mock_event.epoch]
        for metric_name, value in sample_metrics.items():
            assert metric_name in logs
            assert logs[metric_name] == [value]

        tracker.notify(epoch_metrics_mock_event)
        assert epochs == [
            epoch_metrics_mock_event.epoch,
            epoch_metrics_mock_event.epoch,
        ]
        for metric_name, value in sample_metrics.items():
            assert metric_name in logs
            assert logs[metric_name] == [value, value]

    def test_load_model_notification(
        self,
        example_model_name,
        example_sourced_metrics,
        load_model_mock_event,
        tracker,
    ) -> None:
        """Test notification with LoadModel event."""
        tracker.notify(load_model_mock_event)
        example_model_name = load_model_mock_event.model_name
        source_name = set(example_sourced_metrics)
        assert example_model_name in tracker.model_dict
        assert source_name == set(tracker.model_dict[example_model_name])


class TestBasePlotter:
    """Tests for BasePlotter."""

    @pytest.fixture(scope='class', params=[-5, -1, 2, 3])
    def start(self, request) -> int:
        """Instance argument."""
        return request.param

    @pytest.fixture(scope='class', params=['All', 'Selected', 'Skipped'])
    def model_names(self, request, example_model_name) -> tuple[str, ...]:
        """Instance argument."""
        if request.param == 'All':
            return ()

        if request.param == 'Selected':
            return (example_model_name,)

        return ('wrong_name',)

    @pytest.fixture(scope='class', params=['All', 'Selected', 'Skipped'])
    def source_names(self, request, example_sourced_metrics) -> tuple[str, ...]:
        """Instance argument."""
        if request.param == 'All':
            return ()

        if request.param == 'Selected':
            return tuple(example_sourced_metrics)

        return ('wrong_name',)

    @pytest.fixture(scope='class', params=['All', 'Selected', 'Skipped'])
    def metric_names(self, request, example_loss_name) -> tuple[str, ...]:
        """Instance argument."""
        if request.param == 'All':
            return ()

        if request.param == 'Selected':
            return (example_loss_name,)

        return ('wrong_name',)

    @pytest.fixture(scope='class')
    def plotter(
        self, example_model_name, example_sourced_metrics
    ) -> BasePlotter[str]:
        """Set up the instance."""
        instance = _ConcretePlotter()
        instance.model_dict[example_model_name] = example_sourced_metrics
        return instance

    @pytest.fixture(scope='class')
    def plotter_with_loader(self, metric_loader) -> BasePlotter[str]:
        """Expected metrics."""
        return _ConcretePlotter(metric_loader=metric_loader)

    @pytest.fixture(scope='class')
    def plotter_with_start(self, start) -> BasePlotter[str]:
        """Set up the instance."""
        return _ConcretePlotter(start=start)

    @pytest.fixture(scope='class')
    def plotter_with_models(
        self, example_model_name, example_sourced_metrics, model_names
    ) -> BasePlotter[str]:
        """Set up the instance."""
        instance = _ConcretePlotter(model_names=model_names)
        instance.model_dict[example_model_name] = example_sourced_metrics
        instance.model_dict['other_model'] = example_sourced_metrics
        return instance

    def test_end_epoch(
        self, mocker, end_epoch_mock_event, start, plotter_with_start
    ) -> None:
        """Test notification with EndEpoch event."""
        epoch = 4  # defined locally for convenience
        end_epoch_mock_event.epoch = epoch
        example_model_name = end_epoch_mock_event.model_name
        spied = mocker.patch.object(plotter_with_start, '_update_plot')
        plotter_with_start.notify(end_epoch_mock_event)
        if start == -5:  # start + epoch < 0
            true_start = 1
        elif start == -1:  # start + epoch < 3
            true_start = 3
        elif start == 2:  # epoch >= 2 * start
            true_start = 2
        elif start == 3:  # epoch < 2 * start
            true_start = 1
        else:
            raise ValueError('Value not anticipated.')

        spied.assert_called_once_with(
            model_name=example_model_name, start=true_start
        )

    def test_end_test(
        self, mocker, end_test_mock_event, start, plotter_with_start
    ) -> None:
        """Test notification with EndTest event."""
        epoch = 4  # defined locally for convenience
        end_test_mock_event.epoch = epoch
        example_model_name = end_test_mock_event.model_name
        mock_plot = mocker.patch.object(plotter_with_start, '_update_plot')
        plotter_with_start.notify(end_test_mock_event)
        if start == -5:
            true_start = 1  # start < 0
        elif start == -1:
            true_start = 1  # start < 0
        elif start == 2:
            true_start = 2  # start > 0
        elif start == 3:
            true_start = 3  # start > 0
        else:
            raise ValueError('Value not anticipated.')

        mock_plot.assert_called_once_with(
            model_name=example_model_name, start=true_start
        )

    def test_plot_validation(self, example_model_name, start, plotter) -> None:
        """Test validation."""
        if start <= 0:
            with pytest.raises(ValueError):
                plotter.plot(example_model_name, start_epoch=start)
        else:
            with pytest.raises(ValueError):
                plotter.plot('wrong_name', start_epoch=start)

    def test_plot_load(self, example_model_name, plotter_with_loader) -> None:
        """Test loading."""
        assert example_model_name not in plotter_with_loader.model_dict
        plotter_with_loader.plot(example_model_name)
        assert example_model_name in plotter_with_loader.model_dict
        old_metrics = plotter_with_loader.model_dict[example_model_name]
        plotter_with_loader.model_dict[example_model_name] = old_metrics.copy()
        plotter_with_loader.plot(example_model_name)
        new_metrics = plotter_with_loader.model_dict[example_model_name]
        assert old_metrics is not new_metrics  # should load the copy

    def test_select_models(
        self, mocker, example_model_name, model_names, plotter_with_models
    ) -> None:
        """Test _update_plot discards unwanted models."""
        plot_mock = mocker.patch.object(plotter_with_models, '_plot')
        plotter_with_models._update_plot(example_model_name, 1)
        if model_names and model_names[0] != example_model_name:
            plot_mock.assert_not_called()
        else:
            plot_mock.assert_called_once()

    def test_select_sources(
        self,
        mocker,
        example_model_name,
        example_sourced_metrics,
        source_names,
        plotter,
    ) -> None:
        """Test _plot discards unwanted sources."""
        plot_mock = mocker.patch.object(plotter, '_plot_metric')
        plotter._plot(example_model_name, source_names, (), 1)
        if not source_names:
            source_names = tuple(example_sourced_metrics)

        if source_names[0] not in example_sourced_metrics:
            plot_mock.assert_not_called()
        else:
            # empty sources are skipped
            source_names = tuple(
                name for name in source_names if name != 'empty_source'
            )
            # not all metrics have all the sources
            all_called = set[str]()
            for call in plot_mock.call_args_list:
                for name in call.kwargs:
                    all_called.add(name)
            assert all_called == set(source_names)

    def test_select_metrics(
        self,
        mocker,
        example_model_name,
        example_loss_name,
        metric_names,
        plotter,
    ) -> None:
        """Test _plot discards unwanted metrics."""
        plot_mock = mocker.patch.object(plotter, '_plot_metric')
        plotter._plot(example_model_name, (), metric_names, 1)
        if not metric_names:
            assert plot_mock.call_count > 1
        elif metric_names[0] != example_loss_name:
            plot_mock.assert_not_called()
        else:
            assert plot_mock.call_count == 1

    def test_process_source(
        self,
        example_loss_name,
        example_source_name,
        example_sourced_metrics,
        plotter,
    ) -> None:
        """Test all the pre_processing helper functions."""
        # Test _filter_metric
        sourced_metric = plotter._filter_metric(
            example_sourced_metrics, example_loss_name
        )
        assert example_source_name in sourced_metric
        assert 'empty_source' not in sourced_metric

        # Test _order_sources
        ordered_sources = plotter._order_sources(sourced_metric)
        list_ordered = list(ordered_sources)
        list_sourced = list(sourced_metric)
        assert ordered_sources == sourced_metric  # same mapping
        assert list_ordered[2] == list_sourced[0]  # source having fewer epochs
        assert list_ordered[:2] == list_sourced[1:]  # order is kept otherwise

        # Test _source_to_numpy
        sourced_array = plotter._source_to_numpy(ordered_sources)
        array = sourced_array[example_source_name]
        assert array.ndim == 2
        assert array.shape[1] == 2

        # Test _order_sources
        start = example_sourced_metrics[example_source_name][0][-1]
        filtered_sources = plotter._filter_by_epoch(sourced_array, start=start)
        assert filtered_sources.keys() != ordered_sources.keys()
        assert filtered_sources[example_source_name].shape == (1, 2)
        assert filtered_sources[example_source_name][0][0] == start

        # Test _process_source
        pipelined = plotter._process_source(
            example_sourced_metrics, example_loss_name, start=start
        )
        for source_name, array in filtered_sources.items():
            assert np.array_equal(pipelined[source_name], array)

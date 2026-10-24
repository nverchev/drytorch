"""Tests for the ``base_classes`` module."""

import copy
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
    SourcedArray,
    SourcedMetrics,
    TestResults,
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
    def __init__(self, sourced_metrics: SourcedMetrics) -> None:
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


class _ConcretePlotter(BasePlotter[SourcedArray]):
    file_suffix = '.txt'

    def __init__(
        self,
        *,
        metric_loader: MetricLoader | None = None,
        par_dir: pathlib.Path | None = None,
    ) -> None:
        super().__init__(metric_loader=metric_loader, par_dir=par_dir)
        self.rendered: list[list[SourcedArray]] = []
        return

    def _create_training_plot(
        self, model_name: str, metric_name: str, sourced_array: SourcedArray
    ) -> SourcedArray:
        _not_used = model_name, metric_name
        return sourced_array

    def _create_test_plot(
        self, model_name: str, test_results: TestResults
    ) -> SourcedArray:
        _not_used = model_name
        return {
            source_name: np.array(list(metrics.values()))
            for source_name, (_, metrics) in test_results.items()
        }

    def _render(self, figures: list[SourcedArray]) -> None:
        self.rendered.append(figures)
        return

    def _save_plot(self, figure: SourcedArray, path: pathlib.Path) -> None:
        path.write_text(' '.join(figure))
        return


@pytest.fixture
def example_test_results() -> TestResults:
    """Example of the tests of a model."""
    return {
        'Test': (5, {'Accuracy': 0.9, 'Loss': 0.3}),
        'Test_shifted': (5, {'Accuracy': 0.7}),
    }


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

    def test_pause_stashes_and_continue_restores(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that pause stashes state and continue restores it."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)

        tracker.notify(continue_experiment_mock_event)

        assert tracker._run_id == start_experiment_mock_event.run_id
        assert len(tracker._stashed_state) == 0

    def test_pause_start_stop_continue_keeps_state(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        stop_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that concurrent runs do not corrupt stashed state."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)

        start_2 = copy.copy(start_experiment_mock_event)
        start_2.run_id = 'run2'
        tracker.notify(start_2)

        stop_2 = copy.copy(stop_experiment_mock_event)
        stop_2.run_id = 'run2'
        tracker.notify(stop_2)

        tracker.notify(continue_experiment_mock_event)

        assert tracker._run_id == start_experiment_mock_event.run_id

    def test_unrecorded_run_is_not_recorded(
        self, tracker, start_experiment_mock_event
    ) -> None:
        """Test the dumper knows that an unrecorded run is not recorded."""
        start_experiment_mock_event.record = False

        tracker.notify(start_experiment_mock_event)

        assert tracker.record is False

    def test_continued_unrecorded_run_stays_unrecorded(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        stop_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test a recorded run in between keeps a paused one unrecorded."""
        start_experiment_mock_event.record = False
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)
        start_2 = copy.copy(start_experiment_mock_event)
        start_2.run_id = 'run2'
        start_2.record = True
        tracker.notify(start_2)
        stop_2 = copy.copy(stop_experiment_mock_event)
        stop_2.run_id = 'run2'
        tracker.notify(stop_2)
        continue_experiment_mock_event.record = False

        tracker.notify(continue_experiment_mock_event)

        assert tracker.record is False

    def test_continue_without_stash_raises_error(
        self, tracker, continue_experiment_mock_event
    ) -> None:
        """Test that continuing without a stash raises an error."""
        with pytest.raises(exceptions.NoStashedStateError):
            tracker.notify(continue_experiment_mock_event)

    def test_clean_up_leaves_stashed_state(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
    ) -> None:
        """Test that clean_up does not clear stashed paused state."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)

        tracker.clean_up()

        assert len(tracker._stashed_state) == 1

    def test_close_releases_stashes(
        self,
        tracker,
        start_experiment_mock_event,
        pause_experiment_mock_event,
        continue_experiment_mock_event,
    ) -> None:
        """Test that close unconditionally releases stashed resources."""
        tracker.notify(start_experiment_mock_event)
        tracker.notify(pause_experiment_mock_event)

        assert len(tracker._stashed_state) == 1

        tracker.close()

        assert len(tracker._stashed_state) == 0
        with pytest.raises(exceptions.NoStashedStateError):
            tracker.notify(continue_experiment_mock_event)


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
        return MemoryMetrics(metric_loader=metric_loader)

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

    def test_test_metrics_notification(
        self, metric_loader, test_metric_mock_event
    ) -> None:
        """Tests are kept apart from the other metrics."""
        tracker = MemoryMetrics(metric_loader=metric_loader)
        model_name = test_metric_mock_event.model_name
        source_name = test_metric_mock_event.source_name

        tracker.notify(test_metric_mock_event)

        assert tracker.test_dict[model_name][source_name] == (
            test_metric_mock_event.epoch,
            dict(test_metric_mock_event.metrics),
        )
        assert model_name not in tracker.model_dict

    def test_test_metrics_keep_the_latest(
        self, metric_loader, test_metric_mock_event
    ) -> None:
        """A second test of a source replaces the first."""
        tracker = MemoryMetrics(metric_loader=metric_loader)
        model_name = test_metric_mock_event.model_name
        source_name = test_metric_mock_event.source_name
        tracker.notify(test_metric_mock_event)
        test_metric_mock_event.epoch += 1

        tracker.notify(test_metric_mock_event)

        epoch, _ = tracker.test_dict[model_name][source_name]
        assert epoch == test_metric_mock_event.epoch

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

    @pytest.fixture
    def plotter(
        self, example_model_name, example_sourced_metrics
    ) -> _ConcretePlotter:
        """Set up the instance."""
        instance = _ConcretePlotter()
        instance.model_dict[example_model_name] = example_sourced_metrics
        return instance

    @pytest.fixture
    def plotter_with_loader(self, metric_loader) -> _ConcretePlotter:
        """Set up the instance with a metric loader."""
        return _ConcretePlotter(metric_loader=metric_loader)

    def test_create_training_plots_start_epoch_validation(
        self, example_model_name, plotter
    ) -> None:
        """A start epoch less than one raises ValueError."""
        with pytest.raises(ValueError, match='Start epoch'):
            plotter.create_training_plots(example_model_name, start_epoch=0)

    def test_create_training_plots_missing_model(self, plotter) -> None:
        """A model without metrics raises ValueError."""
        with pytest.raises(ValueError, match='No metric to plot'):
            plotter.create_training_plots('wrong_name')

    def test_create_training_plots_load(
        self, example_model_name, plotter_with_loader
    ) -> None:
        """Metrics are loaded when the model has none in memory."""
        plotter_with_loader.create_training_plots(example_model_name)

        assert example_model_name in plotter_with_loader.model_dict

    def test_create_training_plots_keeps_metrics_in_memory(
        self, example_model_name, plotter_with_loader
    ) -> None:
        """Metrics in memory are not loaded again."""
        plotter_with_loader.create_training_plots(example_model_name)
        old_metrics = plotter_with_loader.model_dict[example_model_name]
        copied_metrics = old_metrics.copy()
        plotter_with_loader.model_dict[example_model_name] = copied_metrics

        plotter_with_loader.create_training_plots(example_model_name)

        new_metrics = plotter_with_loader.model_dict[example_model_name]
        assert new_metrics is copied_metrics

    def test_create_training_plots_all_metrics(
        self, example_model_name, example_loss_name, plotter
    ) -> None:
        """Every metric gets its arrays, without the empty sources."""
        figures = plotter.create_training_plots(example_model_name)

        assert set(figures) == {
            example_loss_name,
            'example_metric',
            'example_metric_2',
        }
        assert 'empty_source' not in figures[example_loss_name]

    def test_create_training_plots_selected_metrics(
        self, example_model_name, example_loss_name, plotter
    ) -> None:
        """Only the selected metrics get their arrays."""
        figures = plotter.create_training_plots(
            example_model_name, metric_names=(example_loss_name,)
        )

        assert list(figures) == [example_loss_name]

    def test_create_training_plots_selected_sources(
        self, example_model_name, example_source_name, plotter
    ) -> None:
        """Only the selected sources get their arrays."""
        figures = plotter.create_training_plots(
            example_model_name, source_names=(example_source_name,)
        )

        for sourced_array in figures.values():
            assert list(sourced_array) == [example_source_name]

    @pytest.mark.parametrize(
        'selection',
        [{'metric_names': ('wrong_name',)}, {'source_names': ('wrong_name',)}],
    )
    def test_create_training_plots_nothing_to_plot(
        self, example_model_name, plotter, selection
    ) -> None:
        """A selection that leaves no metric raises ValueError."""
        with pytest.raises(ValueError, match='No metric to plot'):
            plotter.create_training_plots(example_model_name, **selection)

    def test_create_training_plots_does_not_render(
        self, example_model_name, plotter
    ) -> None:
        """Creating the figures does not render them."""
        plotter.create_training_plots(example_model_name)

        assert plotter.rendered == []

    def test_plot_renders(self, example_model_name, plotter) -> None:
        """Plotting renders the figures of all the metrics together."""
        expected = plotter.create_training_plots(example_model_name)

        plotter.plot(example_model_name)

        assert len(plotter.rendered) == 1
        assert len(plotter.rendered[0]) == len(expected)

    def test_save_plots(
        self,
        tmp_path,
        example_model_name,
        example_sourced_metrics,
        start_experiment_mock_event,
    ) -> None:
        """Each figure is saved in a folder of the model inside the run."""
        plotter = _ConcretePlotter(par_dir=tmp_path)
        plotter.model_dict[example_model_name] = example_sourced_metrics
        plotter.notify(start_experiment_mock_event)

        plotter.save_plots(example_model_name)

        saved = sorted(path.name for path in tmp_path.glob('**/*.txt'))
        assert saved == sorted(
            f'{name}.txt'
            for name in plotter.create_training_plots(example_model_name)
        )
        assert all(
            path.parent.name == example_model_name
            for path in tmp_path.glob('**/*.txt')
        )

    def test_create_test_plot_without_tests(
        self, example_model_name, plotter
    ) -> None:
        """A model without tests has no test figure."""
        assert plotter.create_test_plot(example_model_name) is None

    def test_create_test_plot(
        self, example_model_name, plotter, example_test_results
    ) -> None:
        """The test figure holds the tests, filtered by source and metric."""
        plotter.test_dict[example_model_name] = example_test_results

        figure = plotter.create_test_plot(
            example_model_name,
            metric_names=('Accuracy',),
            source_names=('Test',),
        )

        assert figure is not None
        assert list(figure) == ['Test']
        assert list(figure['Test']) == [0.9]

    def test_plot_renders_curves_and_tests(
        self, example_model_name, plotter, example_test_results
    ) -> None:
        """Plotting renders the test figure with the curves."""
        plotter.test_dict[example_model_name] = example_test_results
        n_curves = len(plotter.create_training_plots(example_model_name))

        plotter.plot(example_model_name)

        assert len(plotter.rendered[0]) == n_curves + 1

    def test_plot_tests_only(self, example_test_results) -> None:
        """A model with only tests renders its test figure."""
        plotter = _ConcretePlotter()
        plotter.test_dict['tested_model'] = example_test_results

        plotter.plot('tested_model')

        assert len(plotter.rendered[0]) == 1

    def test_save_plots_tests(
        self,
        tmp_path,
        example_model_name,
        example_sourced_metrics,
        example_test_results,
        start_experiment_mock_event,
    ) -> None:
        """The test figure is saved next to the curves, named tests."""
        plotter = _ConcretePlotter(par_dir=tmp_path)
        plotter.model_dict[example_model_name] = example_sourced_metrics
        plotter.test_dict[example_model_name] = example_test_results
        plotter.notify(start_experiment_mock_event)

        plotter.save_plots(example_model_name)

        saved = {path.name for path in tmp_path.glob('**/*.txt')}
        assert 'tests.txt' in saved

    def test_save_plots_outside_run(self, example_model_name, plotter) -> None:
        """Saving without an active run raises AccessOutsideScopeError."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            plotter.save_plots(example_model_name)

    def test_end_epoch_does_not_plot(self, end_epoch_mock_event) -> None:
        """The end of an epoch does not render figures."""
        plotter = _ConcretePlotter()
        plotter.notify(
            log_events.MetricEvent(
                epoch=1,
                model_name=end_epoch_mock_event.model_name,
                source_name='val',
                metrics={'metric_1': 1.0},
            )
        )

        plotter.notify(end_epoch_mock_event)

        assert plotter.rendered == []

    def test_process_source(
        self,
        example_loss_name,
        example_source_name,
        example_sourced_metrics,
        plotter,
    ) -> None:
        """Test all the pre_processing helper functions."""
        # Test _get_sourced_metric
        sourced_metric = plotter._get_sourced_metric(
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

    def test_process_source_epoch_mismatch_raises_tracker_error(
        self,
        plotter,
    ) -> None:
        """Test mismatched epoch and value list lengths raise TrackerError."""
        sourced_metrics: SourcedMetrics = {
            'val': ([1, 2, 3], {'metric_a': [1.0, 2.0]}),
        }

        with pytest.raises(exceptions.TrackerError) as exc_info:
            plotter._process_source(sourced_metrics, 'metric_a', start=1)

        error_msg = str(exc_info.value)
        assert 'val' in error_msg
        assert 'metric_a' in error_msg

    def test_create_training_plots_missing_initial_epoch_raises_tracker_error(
        self,
    ) -> None:
        """Test metric appearing only from second epoch raises TrackerError."""
        plotter = _ConcretePlotter()
        model_name = 'model_a'
        source_name = 'val'
        event_1 = log_events.MetricEvent(
            epoch=1,
            model_name=model_name,
            source_name=source_name,
            metrics={'metric_1': 1.0},
        )
        event_2 = log_events.MetricEvent(
            epoch=2,
            model_name=model_name,
            source_name=source_name,
            metrics={'metric_1': 2.0, 'metric_2': 5.0},
        )
        plotter.notify(event_1)
        plotter.notify(event_2)

        with pytest.raises(exceptions.TrackerError) as exc_info:
            plotter.create_training_plots(model_name)

        error_msg = str(exc_info.value)
        assert source_name in error_msg
        assert 'metric_2' in error_msg

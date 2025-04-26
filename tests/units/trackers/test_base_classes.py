"""Tests for the base_classes module."""

import pathlib
import pytest
import numpy as np

from dry_torch import exceptions, log_events
from dry_torch.trackers.base_classes import (
    AbstractDumper,
    MetricLoader,
    MemoryMetrics,
    BasePlotter
)

class TestAbstractDumper:
    """Tests for the AbstractDumper."""

    @pytest.fixture(autouse=True)
    @pytest.mark.parametrize('par_dir', [None, 'test'])
    def setup(self, par_dir) -> None:
        """"""
        self.par_dir = par_dir
        self.tracker = AbstractDumper(par_dir=par_dir)

    @pytest.mark.parametrize('par_dir', [None, 'test'])
    def test_par_dir(self,
                     start_experiment_event,
                     stop_experiment_event,
                     par_dir) -> None:
        """"""
        with pytest.raises(RuntimeError):
            _ = self.tracker.par_dir
        self.tracker.notify(start_experiment_event)
        if par_dir is None:
            assert self.tracker.par_dir == start_experiment_event.exp_dir
        else:
            assert self.tracker.par_dir == par_dir
        self.tracker.notify(stop_experiment_event)
        with pytest.raises(RuntimeError):
            _ = self.tracker.par_dir


class ConcreteAbstractDumper(AbstractDumper):
    """Concrete implementation of AbstractDumper for testing."""
    pass


class ConcreteMetricLoader(MetricLoader):
    """Concrete implementation of MetricLoader for testing."""

    def __init__(self):
        super().__init__()
        self.metrics_data = {
            "test_model": {
                "test_source": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3],
                                            "accuracy": [0.7, 0.8, 0.9]})
            }
        }

    def _load_metrics(self, model_name, max_epoch=-1):
        metrics = self.metrics_data.get(model_name, {})
        if max_epoch > 0:
            result = {}
            for source, (epochs, metrics_dict) in metrics.items():
                filtered_epochs = []
                filtered_metrics = {k: [] for k in metrics_dict}

                for i, epoch in enumerate(epochs):
                    if epoch <= max_epoch:
                        filtered_epochs.append(epoch)
                        for metric_name, values in metrics_dict.items():
                            filtered_metrics[metric_name].append(values[i])

                result[source] = (filtered_epochs, filtered_metrics)
            return result
        return metrics


class ConcretePlotter(BasePlotter):
    """Concrete implementation of BasePlotter for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plots = []

    def _plot_metric(self, model_name, metric_name, **sources):
        plot = {
            "model_name": model_name,
            "metric_name": metric_name,
            "sources": sources
        }
        self.plots.append(plot)
        return plot


class TestAbstractDumper:
    """Tests for the AbstractDumper class."""

    def test_initialization(self):
        """Test basic initialization."""
        dumper = ConcreteAbstractDumper()
        assert dumper._par_dir is None
        assert dumper._exp_dir is None

        custom_dir = pathlib.Path("/tmp/test_dir")
        dumper_with_dir = ConcreteAbstractDumper(par_dir=custom_dir)
        assert dumper_with_dir._par_dir == custom_dir

    def test_par_dir_without_exp_dir(self):
        """Test accessing par_dir when both par_dir and exp_dir are None."""
        dumper = ConcreteAbstractDumper()
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _ = dumper.par_dir

    def test_start_experiment_notification(self, start_experiment_event,
                                           tmpdir):
        """Test notification with StartExperiment event."""
        dumper = ConcreteAbstractDumper()
        start_experiment_event.exp_dir = pathlib.Path(tmpdir)
        dumper.notify(start_experiment_event)

        assert dumper._exp_dir == pathlib.Path(tmpdir)
        assert dumper.par_dir == pathlib.Path(tmpdir)

    def test_stop_experiment_notification(self, start_experiment_event,
                                          stop_experiment_event, tmpdir):
        """Test notification with StopExperiment event."""
        dumper = ConcreteAbstractDumper()
        start_experiment_event.exp_dir = pathlib.Path(tmpdir)
        dumper.notify(start_experiment_event)
        dumper.notify(stop_experiment_event)

        assert dumper._exp_dir is None
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _ = dumper.par_dir

    def test_custom_par_dir(self, start_experiment_event, tmpdir):
        """Test using custom par_dir instead of experiment dir."""
        custom_dir = pathlib.Path(tmpdir) / "custom"
        dumper = ConcreteAbstractDumper(par_dir=custom_dir)

        # Even with experiment event, should use custom dir
        start_experiment_event.exp_dir = pathlib.Path(tmpdir) / "experiment"
        dumper.notify(start_experiment_event)

        assert dumper.par_dir == custom_dir
        assert not custom_dir.exists()  # Should be created only when accessed

        # Access to create the directory
        _ = dumper.par_dir
        assert custom_dir.exists()


class TestMetricLoader:
    """Tests for the MetricLoader class."""

    def test_load_metrics_validation(self, mocker):
        """Test validation in load_metrics method."""
        loader = ConcreteMetricLoader()

        # Mock the current experiment to avoid AccessOutsideScopeError
        mocker.patch("dry_torch.experiments.Experiment.current")

        # Test invalid max_epoch
        with pytest.raises(exceptions.TrackerException):
            loader.load_metrics("test_model", max_epoch=-2)

        # Test max_epoch = 0
        result = loader.load_metrics("test_model", max_epoch=0)
        assert result == {}

    def test_load_metrics_without_experiment(self):
        """Test load_metrics when no experiment is active."""
        loader = ConcreteMetricLoader()

        with pytest.raises(exceptions.AccessOutsideScopeError):
            loader.load_metrics("test_model")

    def test_load_metrics_implementation(self, mocker):
        """Test that _load_metrics is called with correct parameters."""
        loader = ConcreteMetricLoader()

        # Mock the current experiment to avoid AccessOutsideScopeError
        mocker.patch("dry_torch.experiments.Experiment.current")

        # Spy on the _load_metrics method
        spy = mocker.spy(loader, "_load_metrics")

        loader.load_metrics("test_model", max_epoch=2)
        spy.assert_called_once_with("test_model", 2)


class TestMemoryMetrics:
    """Tests for the MemoryMetrics class."""

    def test_initialization(self):
        """Test basic initialization."""
        metrics = MemoryMetrics()
        assert metrics.metric_loader is None
        assert metrics.model_metrics == {}

        loader = ConcreteMetricLoader()
        metrics_with_loader = MemoryMetrics(metric_loader=loader)
        assert metrics_with_loader.metric_loader is loader

    def test_metrics_notification(self, sample_metrics):
        """Test notification with Metrics event."""
        metrics_tracker = MemoryMetrics()

        # Create a metrics event
        event = log_events.Metrics(
            model_name="test_model",
            source_name="test_source",
            epoch=1,
            metrics=sample_metrics
        )

        metrics_tracker.notify(event)

        # Check that metrics were stored correctly
        assert "test_model" in metrics_tracker.model_metrics
        assert "test_source" in metrics_tracker.model_metrics["test_model"]

        epochs, logs = metrics_tracker.model_metrics["test_model"][
            "test_source"]
        assert epochs == [1]
        for metric_name, value in sample_metrics.items():
            assert metric_name in logs
            assert logs[metric_name] == [value]

        # Add another epoch
        event2 = log_events.Metrics(
            model_name="test_model",
            source_name="test_source",
            epoch=2,
            metrics=sample_metrics
        )

        metrics_tracker.notify(event2)

        # Check that metrics were appended
        epochs, logs = metrics_tracker.model_metrics["test_model"][
            "test_source"]
        assert epochs == [1, 2]
        for metric_name, value in sample_metrics.items():
            assert logs[metric_name] == [value, value]

    def test_load_model_notification(self, load_model_event):
        """Test notification with LoadModel event when metric_loader is available."""
        loader = ConcreteMetricLoader()
        metrics_tracker = MemoryMetrics(metric_loader=loader)

        # Verify initial state
        assert metrics_tracker.model_metrics == {}

        # Notify with load model event
        metrics_tracker.notify(load_model_event)

        # Verify metrics were loaded from loader
        assert "test_model" in metrics_tracker.model_metrics
        assert "test_source" in metrics_tracker.model_metrics["test_model"]

    def test_load_model_notification_without_loader(self, load_model_event):
        """Test notification with LoadModel event when no metric_loader is available."""
        metrics_tracker = MemoryMetrics()

        # Verify initial state
        assert metrics_tracker.model_metrics == {}

        # Notify with load model event
        metrics_tracker.notify(load_model_event)

        # Verify no change in metrics
        assert metrics_tracker.model_metrics == {}


class TestBasePlotter:
    """Tests for the BasePlotter class."""

    def test_initialization(self):
        """Test basic initialization."""
        plotter = ConcretePlotter()
        assert plotter._model_names == ()
        assert plotter._metric_names == ()
        assert plotter._start == 1
        assert plotter.metric_loader is None

        # With parameters
        loader = ConcreteMetricLoader()
        plotter = ConcretePlotter(
            model_names=["model1", "model2"],
            metric_names=["loss", "accuracy"],
            metric_loader=loader,
            start=5
        )
        assert plotter._model_names == ["model1", "model2"]
        assert plotter._metric_names == ["loss", "accuracy"]
        assert plotter._start == 5
        assert plotter.metric_loader is loader

    def test_end_epoch_notification(self, mocker):
        """Test notification with EndEpoch event."""
        plotter = ConcretePlotter(start=3)

        # Mock _update_plot to check if it's called
        update_mock = mocker.patch.object(plotter, "_update_plot")

        # Create and notify with end epoch event
        event = log_events.EndEpoch(
            source_name="test_source",
            model_name="test_model",
            epoch=5
        )
        plotter.notify(event)

        # Should use start=3 because epoch >= 2*start
        update_mock.assert_called_once_with(model_name="test_model", start=3)

        # Reset mock and test early epoch
        update_mock.reset_mock()
        event = log_events.EndEpoch(
            source_name="test_source",
            model_name="test_model",
            epoch=2
        )
        plotter.notify(event)

        # Should use start=1 because epoch < 2*start
        update_mock.assert_called_once_with(model_name="test_model", start=1)

        # Test negative start
        plotter = ConcretePlotter(start=-2)
        update_mock = mocker.patch.object(plotter, "_update_plot")

        event = log_events.EndEpoch(
            source_name="test_source",
            model_name="test_model",
            epoch=5
        )
        plotter.notify(event)

        # Should use start=3 (epoch + start) because start is negative
        update_mock.assert_called_once_with(model_name="test_model", start=3)

    def test_end_test_notification(self, mocker):
        """Test notification with EndTest event."""
        plotter = ConcretePlotter(start=3)

        # Mock _update_plot to check if it's called
        update_mock = mocker.patch.object(plotter, "_update_plot")

        # Create and notify with end test event
        event = log_events.EndTest(
            source_name="test_source",
            model_name="test_model"
        )
        plotter.notify(event)

        # Should use start=3 for EndTest
        update_mock.assert_called_once_with(model_name="test_model", start=3)

        # Test negative start
        plotter = ConcretePlotter(start=-2)
        update_mock = mocker.patch.object(plotter, "_update_plot")

        plotter.notify(event)

        # Should use start=1 because max(1, -2) = 1
        update_mock.assert_called_once_with(model_name="test_model", start=1)

    def test_update_plot(self, mocker):
        """Test _update_plot method."""
        # Create a plotter with model filter
        plotter = ConcretePlotter(model_names=["test_model"])

        # Set up model metrics
        plotter.model_metrics = {
            "test_model": {
                "source1": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3],
                                        "accuracy": [0.7, 0.8, 0.9]}),
                "source2": ([1, 2], {"loss": [0.6, 0.5]})
            },
            "other_model": {
                "source1": ([1], {"loss": [0.7]})
            }
        }

        # Mock _plot_metric to check calls
        plot_mock = mocker.patch.object(plotter, "_plot_metric")

        # Test with model that passes filter
        plotter._update_plot("test_model", 1)

        # Should be called once for each metric
        assert plot_mock.call_count == 2
        metrics_plotted = {call.kwargs["metric_name"] for call in
                           plot_mock.call_args_list}
        assert metrics_plotted == {"loss", "accuracy"}

        # Reset mock
        plot_mock.reset_mock()

        # Test with model that doesn't pass filter
        plotter._update_plot("other_model", 1)

        # Should not be called
        plot_mock.assert_not_called()

        # Test with metric filter
        plotter = ConcretePlotter(model_names=["test_model"],
                                  metric_names=["loss"])
        plotter.model_metrics = {
            "test_model": {
                "source1": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3],
                                        "accuracy": [0.7, 0.8, 0.9]})
            }
        }

        plot_mock = mocker.patch.object(plotter, "_plot_metric")
        plotter._update_plot("test_model", 1)

        # Should be called once for the filtered metric
        plot_mock.assert_called_once()
        assert plot_mock.call_args.kwargs["metric_name"] == "loss"

    def test_plot(self):
        """Test plot method."""
        loader = ConcreteMetricLoader()
        plotter = ConcretePlotter(metric_loader=loader)

        # Test with invalid start_epoch
        with pytest.raises(ValueError):
            plotter.plot("test_model", start_epoch=0)

        # Test with nonexistent model but with loader
        plots = plotter.plot("test_model")
        assert len(plots) == 2  # loss and accuracy

        # Check plot content
        assert plots[0]["model_name"] == "test_model"
        assert plots[0]["metric_name"] in ["loss", "accuracy"]
        assert "test_source" in plots[0]["sources"]

        # Test with nonexistent model without loader
        plotter = ConcretePlotter()
        with pytest.raises(exceptions.TrackerException):
            plotter.plot("test_model")

        # Test with metrics in memory
        plotter.model_metrics = {
            "test_model": {
                "source1": ([1, 2, 3], {"loss": [0.5, 0.4, 0.3],
                                        "accuracy": [0.7, 0.8, 0.9]}),
                "source2": ([1, 2], {"loss": [0.6, 0.5]})
            }
        }

        # Test with source filter
        plots = plotter.plot("test_model", source_names=["source1"])
        assert len(plots) == 2  # loss and accuracy
        for plot in plots:
            assert list(plot["sources"].keys()) == ["source1"]

        # Test with metric filter
        plots = plotter.plot("test_model", metric_names=["loss"])
        assert len(plots) == 1
        assert plots[0]["metric_name"] == "loss"

        # Test with start_epoch filter
        plots = plotter.plot("test_model", start_epoch=2)
        assert len(plots) == 2

        # Check that epochs are filtered
        for plot in plots:
            for source_data in plot["sources"].values():
                assert all(epoch >= 2 for epoch in source_data[:, 0])

    def test_helper_methods(self):
        """Test the helper methods in BasePlotter."""
        # Test _filter_metric
        source_dict = {
            "source1": (
            [1, 2, 3], {"loss": [0.5, 0.4, 0.3], "accuracy": [0.7, 0.8, 0.9]}),
            "source2": ([1, 2], {"loss": [0.6, 0.5]}),
            "empty": ([], {})
        }

        filtered = BasePlotter._filter_metric(source_dict, "loss")
        assert "source1" in filtered
        assert "source2" in filtered
        assert "empty" not in filtered
        assert filtered["source1"] == ([1, 2, 3], [0.5, 0.4, 0.3])

        # Test _filter_by_epoch
        sources = {
            "source1": np.array([[1, 0.5], [2, 0.4], [3, 0.3]]),
            "source2": np.array([[1, 0.6], [2, 0.5]])
        }

        filtered = BasePlotter._filter_by_epoch(sources, 1)
        assert filtered == sources  # No filtering when start=1

        filtered = BasePlotter._filter_by_epoch(sources, 2)
        assert "source1" in filtered
        assert "source2" in filtered
        assert np.array_equal(filtered["source1"],
                              np.array([[2, 0.4], [3, 0.3]]))
        assert np.array_equal(filtered["source2"], np.array([[2, 0.5]]))

        filtered = BasePlotter._filter_by_epoch(sources, 4)
        assert filtered == {}  # No data left after filtering

        # Test _len_source
        source1 = ("source1", ([1, 2, 3], [0.5, 0.4, 0.3]))
        source2 = ("source2", ([1, 2], [0.6, 0.5]))

        assert BasePlotter._len_source(source1) < BasePlotter._len_source(
            source2)

        # Test _source_to_numpy
        sources = {
            "source1": ([1, 2, 3], [0.5, 0.4, 0.3]),
            "source2": ([1, 2], [0.6, 0.5])
        }

        np_sources = BasePlotter._source_to_numpy(sources)
        assert "source1" in np_sources
        assert "source2" in np_sources
        assert np.array_equal(np_sources["source1"],
                              np.array([[1, 0.5], [2, 0.4], [3, 0.3]]))
        assert np.array_equal(np_sources["source2"],
                              np.array([[1, 0.6], [2, 0.5]]))

    def test_process_source(self):
        """Test the _process_source method."""
        plotter = ConcretePlotter()

        source_dict = {
            "source1": (
            [1, 2, 3], {"loss": [0.5, 0.4, 0.3], "accuracy": [0.7, 0.8, 0.9]}),
            "source2": ([1, 2], {"loss": [0.6, 0.5]}),
            "empty": ([], {})
        }

        processed = plotter._process_source(source_dict, "loss", 1)
        assert "source1" in processed
        assert "source2" in processed
        assert "empty" not in processed

        # Test sorting by length (longest first)
        keys = list(processed.keys())
        assert keys[0] == "source1"
        assert keys[1] == "source2"

        # Test filtering by epoch
        processed = plotter._process_source(source_dict, "loss", 2)
        assert "source1" in processed
        assert "source2" in processed
        assert np.array_equal(processed["source1"],
                              np.array([[2, 0.4], [3, 0.3]]))
        assert np.array_equal(processed["source2"], np.array([[2, 0.5]]))
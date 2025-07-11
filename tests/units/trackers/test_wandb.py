"""Tests for the "wandb" module."""

import pytest
try:
    import wandb
except ImportError:
    pytest.skip('wandb not available', allow_module_level=True)

from typing import Generator

from drytorch import exceptions
from drytorch.trackers.wandb import Wandb


class TestWandb:
    """Tests for the Wandb tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the test environment."""
        self.init_mock = mocker.patch('wandb.init')
        self.finish_mock = mocker.patch('wandb.finish')
        return

    @pytest.fixture
    def tracker(self) -> Wandb:
        """Set up the instance."""
        return Wandb()

    @pytest.fixture
    def tracker_started(
            self,
            tracker,
            start_experiment_mock_event,
            stop_experiment_mock_event,
    ) -> Generator[Wandb, None, None]:
        """Set up started instance."""
        tracker.notify(start_experiment_mock_event)
        yield tracker

        tracker.notify(stop_experiment_mock_event)
        return

    def test_init(self, tracker) -> None:
        """Test initialization."""
        assert isinstance(tracker.resume_run, bool)

    def test_cleanup(self, tracker) -> None:
        tracker.clean_up()
        self.finish_mock.assert_called_once()
        assert tracker._run is None

    def test_notify_start_experiment(self,
                                     mocker,
                                     tracker_started,
                                     start_experiment_mock_event) -> None:
        """Test StartExperiment notification."""
        self.init_mock.assert_called_once_with(
            id=None,
            dir=start_experiment_mock_event.exp_dir,
            project=start_experiment_mock_event.exp_name,
            config=start_experiment_mock_event.config,
            settings=mocker.ANY,
            resume='allow'
        )

    def test_notify_metrics(self,
                            mocker,
                            tracker_started,
                            epoch_metrics_mock_event,
                            example_named_metrics) -> None:
        """Test Metrics notification."""
        log_mock = mocker.patch.object(tracker_started.run, 'log')
        tracker_started.notify(epoch_metrics_mock_event)
        model_name = epoch_metrics_mock_event.model_name
        source_name = epoch_metrics_mock_event.source_name
        expected_metrics = {f'{model_name}/{source_name}-{name}': value
                            for name, value in example_named_metrics.items()}
        log_mock.assert_called_once_with(expected_metrics,
                                         step=epoch_metrics_mock_event.epoch)

    def test_notify_metrics_outside_scope(self,
                                          tracker,
                                          epoch_metrics_mock_event) -> None:
        """Test Metrics notification outside scope."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            tracker.notify(epoch_metrics_mock_event)

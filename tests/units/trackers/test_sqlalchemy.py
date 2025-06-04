"""Tests for the "sqlalchemy" module."""

import pytest
import sqlalchemy
import dataclasses
import datetime
from sqlalchemy import orm, select

from dry_torch.trackers.sqlalchemy import SQLConnection, Log


class TestSQLConnection:
    """Tests for the SQLConnection tracker."""

    @pytest.fixture(autouse=True)
    def setup(self, start_experiment_event) -> None:
        """Setup test environment with in-memory SQLite database."""
        # Use in-memory SQLite for testing
        self.tracker = SQLConnection(drivername='sqlite', database=':memory:')
        self.tracker.notify(start_experiment_event)
        self.exp_name = start_experiment_event.exp_name

    @pytest.fixture
    def epoch_metrics_events(self, epoch_metrics_event, sample_metrics):
        """Create a list of epoch metrics events with different epochs."""
        events = [epoch_metrics_event]

        second_event = dataclasses.replace(epoch_metrics_event)
        second_event.epoch += 1
        events.append(second_event)

        return events

    def test_notify_with_epoch_metrics(self, epoch_metrics_event,
                                       sample_metrics):
        """Test that metrics are correctly stored in the database."""
        # Run the method
        self.tracker.notify(epoch_metrics_event)

        # Check records in the database
        with self.tracker.Session() as session:
            records = session.execute(
                select(Log).where(
                    Log.experiment == self.exp_name,
                    Log.model == epoch_metrics_event.model_name,
                    Log.source == epoch_metrics_event.source,
                    Log.epoch == epoch_metrics_event.epoch
                )
            ).scalars().all()

            # Check if we have the correct number of metrics
            assert len(records) == len(sample_metrics)

            # Check if each metric is correctly stored
            stored_metrics = {record.metric: record.value
                              for record in records}
            for metric_name, expected_value in sample_metrics.items():
                assert metric_name in stored_metrics
                assert stored_metrics[metric_name] == expected_value

    def test_multiple_epochs(self, epoch_metrics_events):
        """Test that multiple epochs are correctly stored."""
        # Add all events
        num_events = len(epoch_metrics_events)
        for event in epoch_metrics_events:
            self.tracker.notify(event)

        # Check records for each epoch
        with self.tracker.Session() as session:
            for event in epoch_metrics_events:
                records = session.execute(
                    select(Log).where(
                        Log.experiment == self.exp_name,
                        Log.model == event.model_name,
                        Log.source == event.source,
                    )
                ).scalars().all()

                # Check correct number of metrics for this epoch
                assert len(records) == num_events * len(event.metrics)
        print(records)

    def test_experiment_scope(self, epoch_metrics_event, stop_experiment_event):
        """Test that metrics cannot be added outside experiment scope."""
        # Stop the experiment
        self.tracker.notify(stop_experiment_event)

        # Attempt to add metrics should raise an error
        with pytest.raises(RuntimeError):
            self.tracker.notify(epoch_metrics_event)

    def test_get_metric(self, epoch_metrics_event):
        """Test get_metric method."""
        self.tracker.notify(epoch_metrics_event)

        for metric_name, value in epoch_metrics_event.metrics.items():
            epochs, values = self.tracker._get_run_metrics(
                model_name=epoch_metrics_event.model_name,
                source=epoch_metrics_event.source,
                metric_name=metric_name,
                exp_name=self.exp_name,
            )
            assert epochs == [epoch_metrics_event.epoch]
            assert values == [value]

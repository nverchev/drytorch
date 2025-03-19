"""Test SQL connection tracker"""

import pytest
import sqlalchemy
import dataclasses
import datetime
from sqlalchemy import orm, select

from dry_torch.trackers.sqlalchemy_backend import SQLConnection, LoggedMetrics


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
                select(LoggedMetrics).where(
                    LoggedMetrics.experiment == self.exp_name,
                    LoggedMetrics.model_name == epoch_metrics_event.model_name,
                    LoggedMetrics.source == epoch_metrics_event.source,
                    LoggedMetrics.epoch == epoch_metrics_event.epoch
                )
            ).scalars().all()

            # Check if we have the correct number of metrics
            assert len(records) == len(sample_metrics)

            # Check if each metric is correctly stored
            stored_metrics = {record.metric_name: record.metric_value
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
                    select(LoggedMetrics).where(
                        LoggedMetrics.experiment == self.exp_name,
                        LoggedMetrics.model_name == event.model_name,
                        LoggedMetrics.source == event.source,
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

    def test_get_metrics(self, epoch_metrics_event):
        self.tracker.notify(epoch_metrics_event)

        # Get metrics for the experiment
        epochs, metrics = self.tracker.get_metrics(
            model_name=epoch_metrics_event.model_name,
            source=epoch_metrics_event.source,
            exp_name=self.exp_name,
        )
        assert epochs == [epoch_metrics_event.epoch]
        assert metrics == {k: [v] for k, v in
                           epoch_metrics_event.metrics.items()}
    #
    # def test_create_pivoted_view(self, epoch_metrics_events):
    #     """Test creating and querying a pivoted view of metrics."""
    #     # This is an implementation for the feature you wanted to add
    #     # Add multiple events
    #     for event in epoch_metrics_events:
    #         self.tracker.notify(event)
    #
    #     # Implement a create_pivoted_view method in the tracker
    #     # This is just a simple implementation for testing purposes
    #     with self.tracker.Session() as session:
    #         # Get distinct metric names for this experiment
    #         metric_names = [row[0] for row in session.query(
    #             LoggedMetrics.metric_name
    #         ).filter(
    #             LoggedMetrics.experiment == self.exp_name
    #         ).distinct()]
    #
    #         # Build case statements for each metric
    #         case_statements = []
    #         for metric_name in metric_names:
    #             case_stmt = sqlalchemy.case(
    #                 [(LoggedMetrics.metric_name == metric_name,
    #                   LoggedMetrics.metric_value)],
    #                 else_=None
    #             ).label(f"metric_{metric_name}")
    #             case_statements.append(case_stmt)
    #
    #         # Create a select statement for the view
    #         query = select(
    #             LoggedMetrics.experiment,
    #             LoggedMetrics.model_name,
    #             LoggedMetrics.source,
    #             LoggedMetrics.epoch,
    #             *case_statements
    #         ).where(
    #             LoggedMetrics.experiment == self.exp_name
    #         ).group_by(
    #             LoggedMetrics.experiment,
    #             LoggedMetrics.model_name,
    #             LoggedMetrics.source,
    #             LoggedMetrics.epoch
    #         )
    #
    #         # Execute the query
    #         results = session.execute(query).all()
    #
    #         # Verify results
    #         assert len(results) == len(
    #             set(event.epoch for event in epoch_metrics_events))
    #
    #         # Check if all epochs are present
    #         epochs = [result.epoch for result in results]
    #         for event in epoch_metrics_events:
    #             assert event.epoch in epochs
    #
    #         # Check if metrics are correctly pivoted
    #         for result in results:
    #             # Find the corresponding event
    #             event = next((e for e in epoch_metrics_events if
    #                           e.epoch == result.epoch), None)
    #             assert event is not None
    #
    #             # Check each metric
    #             for metric_name in event.metrics:
    #                 pivoted_value = getattr(result, f"metric_{metric_name}")
    #                 assert pivoted_value == event.metrics[metric_name]

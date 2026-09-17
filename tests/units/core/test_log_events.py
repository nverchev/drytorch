"""Tests for the ``log_events`` module."""

import dataclasses

import pytest

from drytorch.core import exceptions
from drytorch.core.log_events import Event, IterateBatchEvent


@dataclasses.dataclass(frozen=True)
class _SimpleEvent(Event):
    """Simple Event subclass for testing."""


class TestEvent:
    """Tests for Event."""

    def test_no_auto_publish(self) -> None:
        """Test the error raises correctly when instantiating the class."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _SimpleEvent()


class TestIterateBatchEvent:
    """Tests for IterateBatchEvent."""

    def test_update_and_refresh(self, mocker) -> None:
        """Test update and refresh callback dispatching."""
        Event.set_auto_publish(lambda _: None)
        try:
            event = IterateBatchEvent(
                source_name='test',
                batch_size=32,
                n_iter=10,
                dataset_size=320,
            )
            callback = mocker.Mock()
            event.push_updates.append(callback)
            metrics = {'loss': 0.5}

            event.update(metrics, n_processes=2)

            callback.assert_called_once_with(metrics, 2)
        finally:
            Event.set_auto_publish(None)

    def test_refresh(self, mocker) -> None:
        """Test refresh callback dispatching."""
        Event.set_auto_publish(lambda _: None)
        try:
            event = IterateBatchEvent(
                source_name='test',
                batch_size=32,
                n_iter=10,
                dataset_size=320,
            )
            callback = mocker.Mock()
            event.push_updates.append(callback)
            metrics = {'loss': 0.5}

            event.refresh(metrics)

            callback.assert_called_once_with(metrics, 0)
        finally:
            Event.set_auto_publish(None)

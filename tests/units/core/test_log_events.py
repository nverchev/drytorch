"""Test for the "log_events" module."""

import dataclasses

import pytest

from drytorch.core import exceptions, log_events


@dataclasses.dataclass(frozen=True)
class _SimpleEvent(log_events.Event):
    """Simple Event subclass for testing."""


class TestEvent:
    """Tests for Event."""

    def test_no_auto_publish(self):
        """Test the error raises correctly when instantiating the class."""
        with pytest.raises(exceptions.AccessOutsideScopeError):
            _SimpleEvent()

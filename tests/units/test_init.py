"""Test functions for default trackers."""

import dry_torch


def test_standard_trackers():
    """Test adding standard trackers."""
    dry_torch.remove_all_default_trackers()
    assert not dry_torch.DEFAULT_TRACKERS
    dry_torch.add_standard_trackers_to_default_trackers()
    assert dry_torch.DEFAULT_TRACKERS


def test_remove_trackers():
    """Test removing all default trackers."""
    dry_torch.add_standard_trackers_to_default_trackers()
    assert dry_torch.DEFAULT_TRACKERS
    dry_torch.remove_all_default_trackers()
    assert not dry_torch.DEFAULT_TRACKERS

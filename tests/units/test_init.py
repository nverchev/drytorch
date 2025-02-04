"""Test functions for default trackers."""
import pytest

import importlib

import dry_torch
from dry_torch import FailedOptionalImportWarning


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


def test_failed_import_warning():
    """Test optional import failure raises warning."""
    with pytest.MonkeyPatch().context() as mp:
        original_import = __import__

        def _mock_import(name: str, *args, **kwargs):
            if name == 'tqdm':
                raise ImportError
            if name == 'yaml':
                raise ModuleNotFoundError
            return original_import(name, *args, **kwargs)

        mp.setattr('builtins.__import__', _mock_import)

        with pytest.warns(FailedOptionalImportWarning) as warning_info:
            importlib.reload(dry_torch)

    warnings = [str(w.message) for w in warning_info]
    assert any('tqdm' in w for w in warnings)
    assert any('yaml' in w for w in warnings)

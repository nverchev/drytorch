"""Test functions for default trackers."""
import pytest

import importlib

import dry_torch
from dry_torch.tracking import DEFAULT_TRACKERS
from dry_torch import FailedOptionalImportWarning


def test_standard_trackers():
    """Test setting standard trackers adds trackers."""
    dry_torch.set_standard_trackers()
    assert DEFAULT_TRACKERS


def test_tuning_trackers():
    """Test setting tuning trackers adds trackers."""
    dry_torch.set_tuning_trackers()
    assert DEFAULT_TRACKERS


def test_remove_trackers():
    """Test removing all default trackers."""
    assert DEFAULT_TRACKERS
    dry_torch.remove_all_default_trackers()
    assert not DEFAULT_TRACKERS


def test_failed_import_warning():
    """Test optional import failure raises warning."""
    with pytest.MonkeyPatch().context() as mp:
        original_import = __import__

        def _mock_import(name: str,
                         globals=None,
                         locals=None,
                         fromlist=(),
                         level=0):
            if name == 'dry_torch.trackers' and fromlist:
                if 'tqdm' in fromlist:
                    raise ImportError()
                if 'yaml' in fromlist:
                    raise ModuleNotFoundError()

            return original_import(name, globals, locals, fromlist, level)

        mp.setattr('builtins.__import__', _mock_import)

        with pytest.warns(FailedOptionalImportWarning) as warning_info:
            importlib.reload(dry_torch)

    warnings = [str(w.message) for w in warning_info]
    assert any('tqdm' in w for w in warnings)
    assert any('yaml' in w for w in warnings)

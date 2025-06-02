"""Test functions for default trackers."""
import pytest

import importlib
import os

import dry_torch
from dry_torch.tracking import DEFAULT_TRACKERS
from dry_torch import FailedOptionalImportWarning


def test_standard_trackers():
    """Test setting standard trackers adds trackers."""
    dry_torch.initialize_trackers()
    assert DEFAULT_TRACKERS


def test_env_trackers():
    """Test setting standard trackers adds trackers."""
    dry_torch.remove_all_default_trackers()
    os.environ['DRY_TORCH_INIT_MODE'] = 'none'
    importlib.reload(dry_torch)
    assert not DEFAULT_TRACKERS
    os.environ['DRY_TORCH_INIT_MODE'] = 'hydra'
    importlib.reload(dry_torch)
    assert DEFAULT_TRACKERS


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
            os.environ['DRY_TORCH_INIT_MODE'] = 'standard'
            importlib.reload(dry_torch)

    warnings = [str(w.message) for w in warning_info]
    assert any('tqdm' in w for w in warnings)
    assert any('yaml' in w for w in warnings)

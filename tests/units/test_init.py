"""Tests for the "__init__" module."""

import importlib
import os

import pytest

import drytorch

from drytorch import FailedOptionalImportWarning
from drytorch.core.track import DEFAULT_TRACKERS


def test_standard_trackers():
    """Test setting standard trackers adds trackers."""
    drytorch.init_trackers()
    assert DEFAULT_TRACKERS


def test_env_trackers():
    """Test setting standard trackers adds trackers."""
    drytorch.remove_all_default_trackers()
    os.environ['DRYTORCH_INIT_MODE'] = 'none'
    importlib.reload(drytorch)
    assert not DEFAULT_TRACKERS
    os.environ['DRYTORCH_INIT_MODE'] = 'hydra'
    importlib.reload(drytorch)
    assert DEFAULT_TRACKERS


def test_failed_import_warning():
    """Test optional import failure raises warning."""
    with pytest.MonkeyPatch().context() as mp:
        original_import = __import__

        def _mock_import(
            name: str, globals_=None, locals_=None, fromlist=(), level=0
        ):
            if name == 'drytorch.trackers' and fromlist:
                if 'tqdm' in fromlist:
                    raise ImportError()
                if 'yaml' in fromlist:
                    raise ModuleNotFoundError()

            return original_import(name, globals_, locals_, fromlist, level)

        mp.setattr('builtins.__import__', _mock_import)

        with pytest.warns(FailedOptionalImportWarning) as warning_info:
            os.environ['drytorch_INIT_MODE'] = 'standard'
            importlib.reload(drytorch)

    warnings = [str(w.message) for w in warning_info]
    assert any('tqdm' in w for w in warnings)
    assert any('yaml' in w for w in warnings)

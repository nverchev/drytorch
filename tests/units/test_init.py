"""Tests for the ``__init__`` module."""

import importlib
import os
import warnings

import pytest

import drytorch

from drytorch import FailedOptionalImportWarning
from drytorch.core import exceptions, tracking


def test_standard_trackers() -> None:
    """Test setting standard trackers adds trackers."""
    drytorch.init_trackers()

    assert tracking.DEFAULT_TRACKERS


def test_env_trackers_none() -> None:
    """Test setting DRYTORCH_INIT_MODE to none disables default trackers."""
    drytorch.remove_all_default_trackers()
    os.environ['DRYTORCH_INIT_MODE'] = 'none'

    importlib.reload(drytorch)

    assert not tracking.DEFAULT_TRACKERS


def test_env_trackers_hydra() -> None:
    """Test setting DRYTORCH_INIT_MODE to hydra adds default trackers."""
    drytorch.remove_all_default_trackers()
    os.environ['DRYTORCH_INIT_MODE'] = 'hydra'

    importlib.reload(drytorch)

    assert tracking.DEFAULT_TRACKERS


def test_failed_import_warning() -> None:
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


def test_env_strict_mode_invalid() -> None:
    """Test setting DRYTORCH_STRICT to invalid value raises ValueError."""
    os.environ['DRYTORCH_STRICT'] = '2'
    try:
        with pytest.raises(ValueError, match='DRYTORCH_STRICT'):
            importlib.reload(drytorch)
    finally:
        os.environ.pop('DRYTORCH_STRICT', None)
        importlib.reload(drytorch)


def test_env_strict_mode_enabled() -> None:
    """Test setting DRYTORCH_STRICT to 1 filters warnings as errors."""
    os.environ['DRYTORCH_STRICT'] = '1'
    try:
        importlib.reload(drytorch)
        with pytest.raises(exceptions.UncommittedChangesWarning):
            warnings.warn(exceptions.UncommittedChangesWarning(1), stacklevel=1)
        with pytest.raises(exceptions.GitUnavailableWarning):
            warnings.warn(exceptions.GitUnavailableWarning(), stacklevel=1)
    finally:
        os.environ.pop('DRYTORCH_STRICT', None)
        importlib.reload(drytorch)


def test_git_unavailable_warning_ignored_by_default() -> None:
    """Test GitUnavailableWarning is ignored by default filter."""
    importlib.reload(drytorch)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.warn(exceptions.GitUnavailableWarning(), stacklevel=1)

    assert not any(
        isinstance(w.message, exceptions.GitUnavailableWarning)
        for w in recorded
    )

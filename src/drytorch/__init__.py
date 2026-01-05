"""Init file for the drytorch package.

It automatically initializes some trackers with sets of settings (modes) that
work well together. The mode can be set as an environmental variable
DRYTORCH_INIT_MODE before loading the package or explicitly reset after.

Available modes:
    1) standard: log to stderr, preferring tqdm over the built-in logger.
    2) hydra: log to stdout and accommodate default Hydra settings.
    3) minimal: reduce output and avoid dumping metadata.
    4) none: no tracker is added to the default ones.

Attributes:
    INIT_MODE: the mode the trackers will be initialized with at the start.
        If DRYTORCH_INIT_MODE is not present, it defaults to "standard".
"""

import logging
import os
import sys
import warnings

from importlib.metadata import version
from typing import Literal, TypeGuard

from drytorch.core.exceptions import FailedOptionalImportWarning
from drytorch.core.experiment import Experiment
from drytorch.core.track import (
    Tracker,
    extend_default_trackers,
    remove_all_default_trackers,
)
from drytorch.lib.evaluations import Diagnostic, Test, Validation
from drytorch.lib.learn import LearningSchema
from drytorch.lib.load import DataLoader
from drytorch.lib.models import Model
from drytorch.lib.objectives import Loss, Metric
from drytorch.lib.train import Trainer
from drytorch.trackers import logging as builtin_logging
from drytorch.trackers.logging import INFO_LEVELS


__version__ = version('drytorch')

__all__ = [
    'DataLoader',
    'Diagnostic',
    'Experiment',
    'LearningSchema',
    'Loss',
    'Metric',
    'Model',
    'Test',
    'Trainer',
    'Validation',
    'extend_default_trackers',
    'init_trackers',
    'remove_all_default_trackers',
]

_InitMode = Literal['standard', 'hydra', 'minimal']

logger = logging.getLogger('drytorch')


def init_trackers(mode: _InitMode = 'standard') -> None:
    """Initialize trackers used by default during the experiment.

    Three initializations are available:
        1) standard: log to stderr, preferring tqdm over the built-in logger.
        2) hydra: log to stdout and accommodate default Hydra settings.
        3) minimal: reduce output and avoid dumping metadata.

    Args:
        mode: one of the suggested initialization modes.

    Raises:
        ValueError if mode is not available.
    """
    remove_all_default_trackers()
    tracker_list: list[Tracker] = [builtin_logging.BuiltinLogger()]
    is_tqdm_installed = _add_tqdm(tracker_list, mode=mode)
    if mode == 'hydra':
        # hydra logs to stdout by default
        builtin_logging.enable_default_handler(sys.stdout)
        builtin_logging.enable_propagation()
        _add_yaml(tracker_list)
        verbosity = builtin_logging.INFO_LEVELS.metrics

    elif mode == 'minimal':
        if is_tqdm_installed:
            verbosity = builtin_logging.INFO_LEVELS.training
        else:
            verbosity = builtin_logging.INFO_LEVELS.epoch
            builtin_logging.set_formatter('progress')

    elif mode == 'standard':
        _add_yaml(tracker_list)
        if is_tqdm_installed:
            verbosity = builtin_logging.INFO_LEVELS.epoch
        else:
            verbosity = builtin_logging.INFO_LEVELS.metrics

    else:
        raise ValueError('Mode {mode} not available.')

    extend_default_trackers(tracker_list)
    builtin_logging.set_verbosity(verbosity)
    return


def _add_tqdm(tracker_list: list[Tracker], mode: _InitMode) -> bool:
    try:
        from drytorch.trackers import tqdm
    except (ImportError, ModuleNotFoundError):
        warnings.warn(FailedOptionalImportWarning('tqdm'), stacklevel=2)
        return False

    if mode == 'hydra':
        # progress bar disappears leaving only log metrics.
        tqdm_logger = tqdm.TqdmLogger(leave=False)
    elif mode == 'minimal':
        # double bar replaces most logs.
        tqdm_logger = tqdm.TqdmLogger(enable_training_bar=True)
    else:
        # console metrics from the progress bar
        tqdm_logger = tqdm.TqdmLogger()

    tracker_list.append(tqdm_logger)
    return True


def _add_yaml(tracker_list: list[Tracker]) -> bool:
    try:
        from drytorch.trackers import yaml
    except (ImportError, ModuleNotFoundError):
        warnings.warn(FailedOptionalImportWarning('yaml'), stacklevel=2)
        return False

    tracker_list.append(yaml.YamlDumper())
    return True


def _check_mode_is_valid(
    mode: str,
) -> TypeGuard[Literal['standard', 'hydra', 'minimal']]:
    return mode in ('standard', 'hydra', 'minimal')


INIT_MODE = os.getenv('DRYTORCH_INIT_MODE', 'standard')
if _check_mode_is_valid(INIT_MODE):
    logger.log(INFO_LEVELS.internal, 'Initializing %s mode.', INIT_MODE)
    init_trackers(INIT_MODE)
elif INIT_MODE != 'none':
    raise ValueError(f'DRYTORCH_INIT_MODE: {INIT_MODE} not a valid setting.')

"""Init file for the drytorch package.

It automatically initializes some trackers with sets of settings (modes) that
work well together. The mode can be set as an environmental variable
DRYTORCH_INIT_MODE before loading the package or explicitly reset after.

Available modes:
    1) standard: if present, relies on tqdm to print the metrics on stderr.
    2) hydra: logs metrics to stdout to accommodate default hydra settings.
    3) tuning: most output gets overwritten and metadata is not extracted.
    4) none: skip initialization.

Attributes:
    INIT_MODE: the mode the trackers will be initialized with at the start.
        If DRYTORCH_INIT_MODE is not present it defaults to standard.
"""

import logging
import os
import sys
import warnings

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
    'init_trackers',
]

_InitMode = Literal['standard', 'hydra', 'tuning']

logger = logging.getLogger('drytorch')


def init_trackers(mode: _InitMode = 'standard') -> None:
    """Initialize trackers used by default during the experiment.

    Three initializations are available:
        1) standard: if present, relies on tqdm to print the metrics on stderr.
        2) hydra: logs metrics to stdout to accommodate default hydra settings.
        3) tuning: most output gets overwritten and metadata is not extracted.

    Args:
        mode: one of the suggested initialization modes.

    Raises:
        ValueError if mode is not available.
    """
    remove_all_default_trackers()
    if mode == 'hydra':
        # hydra logs to stdout by default
        builtin_logging.enable_default_handler(sys.stdout)
        builtin_logging.enable_propagation()

    tracker_list: list[Tracker] = [builtin_logging.BuiltinLogger()]
    _add_tqdm(tracker_list, mode=mode)
    if mode != 'tuning':
        _add_yaml(tracker_list)
    extend_default_trackers(tracker_list)
    return


def _add_tqdm(tracker_list: list[Tracker], mode: _InitMode) -> None:
    verbosity = builtin_logging.INFO_LEVELS.metrics

    try:
        from drytorch.trackers import tqdm
    except (ImportError, ModuleNotFoundError):
        warnings.warn(FailedOptionalImportWarning('tqdm'), stacklevel=2)
        if mode == 'tuning':
            verbosity = builtin_logging.INFO_LEVELS.epoch
            builtin_logging.set_formatter('progress')
    else:
        if mode == 'standard':
            # metrics logs redundant because already visible in the progress bar
            verbosity = builtin_logging.INFO_LEVELS.epoch
            tqdm_logger = tqdm.TqdmLogger()
        elif mode == 'tuning':
            # double bar replaces most logs.
            verbosity = builtin_logging.INFO_LEVELS.training
            tqdm_logger = tqdm.TqdmLogger(enable_training_bar=True)
        elif mode == 'hydra':
            # progress bar disappears leaving only log metrics.
            tqdm_logger = tqdm.TqdmLogger(leave=False)
        else:
            raise ValueError('Mode {mode} not available.')

        tracker_list.append(tqdm_logger)
        builtin_logging.set_verbosity(verbosity)
        return


def _add_yaml(tracker_list: list[Tracker]) -> None:
    try:
        from drytorch.trackers import yaml
    except (ImportError, ModuleNotFoundError):
        warnings.warn(FailedOptionalImportWarning('yaml'), stacklevel=2)
    else:
        tracker_list.append(yaml.YamlDumper())


def _check_mode_is_valid(
    mode: str,
) -> TypeGuard[Literal['standard', 'hydra', 'tuning']]:
    return mode in ('standard', 'hydra', 'tuning')


INIT_MODE = os.getenv('DRYTORCH_INIT_MODE', 'standard')
if _check_mode_is_valid(INIT_MODE):
    logger.log(INFO_LEVELS.internal, 'Initializing %s mode.', INIT_MODE)
    init_trackers(INIT_MODE)
elif INIT_MODE != 'none':
    raise ValueError(f'DRYTORCH_INIT_MODE: {INIT_MODE} not a valid setting.')

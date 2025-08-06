"""Init file for the drytorch package.

It automatically initializes some trackers with sets of settings (modes) that
work well together. The mode can be set as an environmental variable before
loading the package or explicitly reset after. Alternatively, skip or remove
initialization and use custom settings.

Allowed values for the environmental variable drytorch_INIT_MODE:
    1) standard: if present, relies on tqdm to print the metrics on stderr.
    2) hydra: logs metrics to stdout to accommodate default hydra settings.
    3) tuning: most output gets overwritten and metadata is not extracted.
    4) none: skip initialization.

Default uses standard mode.
"""

import logging
import os
import sys
import warnings

from typing import Literal, TypeGuard

from drytorch.evaluating import Diagnostic, Test, Validation
from drytorch.exceptions import FailedOptionalImportWarning
from drytorch.experiments import Experiment, SpecsMixin
from drytorch.learning import LearningScheme
from drytorch.loading import DataLoader
from drytorch.metrics import Loss, Metric
from drytorch.models import Model
from drytorch.trackers import logging as builtin_logging
from drytorch.trackers.logging import INFO_LEVELS
from drytorch.tracking import (
    Tracker,
    extend_default_trackers,
    remove_all_default_trackers,
)
from drytorch.training import Trainer


try:
    from drytorch.trackers import yaml
except (ImportError, ModuleNotFoundError) as ie:
    yaml = None
    YAML_EXCEPTION: Exception | None = ie
else:
    YAML_EXCEPTION = None


try:
    from drytorch.trackers import tqdm
except (ImportError, ModuleNotFoundError) as ie:
    tqdm = None
    TQDM_EXCEPTION: Exception | None = ie
else:
    TQDM_EXCEPTION = None

__all__ = [
    'DataLoader',
    'Diagnostic',
    'Experiment',
    'FailedOptionalImportWarning',
    'LearningScheme',
    'Loss',
    'Metric',
    'Model',
    'SpecsMixin',
    'Test',
    'Tracker',
    'Trainer',
    'Validation',
]

logger = logging.getLogger('drytorch')


def initialize_trackers(
    mode: Literal['standard', 'hydra', 'tuning'] = 'standard',
) -> None:
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
    verbosity = builtin_logging.INFO_LEVELS.metrics
    if mode == 'hydra':
        # hydra logs to stdout by default
        builtin_logging.enable_default_handler(sys.stdout)
        builtin_logging.enable_propagation()

    tracker_list: list[Tracker] = [builtin_logging.BuiltinLogger()]

    if tqdm is not None:
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
    else:
        warnings.warn(FailedOptionalImportWarning('tqdm'), stacklevel=2)
        if mode == 'tuning':
            verbosity = builtin_logging.INFO_LEVELS.epoch
            builtin_logging.set_formatter('progress')

    if mode != 'tuning':
        if yaml is not None:
            tracker_list.append(yaml.YamlDumper())
        else:
            warnings.warn(FailedOptionalImportWarning('yaml'), stacklevel=2)

    extend_default_trackers(tracker_list)
    builtin_logging.set_verbosity(verbosity)
    return


def _check_mode_is_valid(
    mode: str,
) -> TypeGuard[Literal['standard', 'hydra', 'tuning']]:
    return mode in ('standard', 'hydra', 'tuning')


init_mode = os.getenv('drytorch_INIT_MODE', 'standard')
if _check_mode_is_valid(init_mode):
    logger.log(INFO_LEVELS.internal, 'Initializing %s mode.', init_mode)
    initialize_trackers(init_mode)
elif init_mode != 'none':
    raise ValueError(f'drytorch_INIT_MODE: {init_mode} not a valid setting.')

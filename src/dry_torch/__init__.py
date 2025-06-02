"""
Init file for the dry_torch package.

It automatically initializes some trackers with sets of settings (modes) that
work well together. The mode can be set as an environmental variable before
loading the package or explicitly reset after. Alternatively, skip or remove
initialization and use custom settings.

Allowed values for the environmental variable DRY_TORCH_INIT_MODE:
    1) standard: if present, relies on tqdm to print the metrics on stderr.
    2) hydra: logs metrics to stdout to accommodate default hydra settings.
    3) tuning: most output gets overwritten and metadata is not extracted.
    4) none: skip initialization.

Default uses standard mode.
"""

import logging
import os
import sys
from typing import Literal, TypeGuard
import warnings

from dry_torch.evaluating import Diagnostic
from dry_torch.evaluating import Test
from dry_torch.evaluating import Validation
from dry_torch.exceptions import FailedOptionalImportWarning
from dry_torch.experiments import Experiment
from dry_torch.experiments import MainExperiment
from dry_torch.experiments import SubExperiment
from dry_torch.metrics import Loss
from dry_torch.metrics import Metric
from dry_torch.loading import DataLoader
from dry_torch.learning import LearningScheme
from dry_torch.learning import Model
from dry_torch.trackers.logging import INFO_LEVELS
from dry_torch.tracking import remove_all_default_trackers
from dry_torch.tracking import extend_default_trackers
from dry_torch.tracking import Tracker
from dry_torch.training import Trainer
from dry_torch.trackers import logging as builtin_logging
from dry_torch.trackers import csv as builtin_csv

logger = logging.getLogger('dry_torch')


def initialize_trackers(
        mode: Literal['standard', 'hydra', 'tuning'] = 'standard',
) -> None:
    """
    Initialize trackers used by default during the experiment.

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
    try:
        from dry_torch.trackers import tqdm

    except (ImportError, ModuleNotFoundError) as ie:
        warnings.warn(FailedOptionalImportWarning('tqdm', ie))
        if mode == 'tuning':
            verbosity = builtin_logging.INFO_LEVELS.epoch
            builtin_logging.set_formatter('progress')

    else:
        if mode == 'standard':
            # metrics logs redundant because already visible in progress bar.
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

    if mode != 'tuning':
        try:
            from dry_torch.trackers import yaml
        except (ImportError, ModuleNotFoundError) as ie:
            warnings.warn(FailedOptionalImportWarning('yaml', ie))
        else:
            tracker_list.append(yaml.YamlDumper())

    extend_default_trackers(tracker_list)
    builtin_logging.set_verbosity(verbosity)
    return


def _check_mode_is_valid(
        mode: str,
) -> TypeGuard[Literal['standard', 'hydra', 'tuning']]:
    return mode in ('standard', 'hydra', 'tuning')


init_mode = os.getenv('DRY_TORCH_INIT_MODE', 'standard')
if _check_mode_is_valid(init_mode):
    logger.log(INFO_LEVELS.internal, f'Initializing %s mode.', init_mode)
    initialize_trackers(init_mode)
elif init_mode != 'none':
    raise ValueError(f'DRY_TORCH_INIT_MODE: {init_mode} not a valid setting.')

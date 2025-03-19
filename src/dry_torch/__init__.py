"""Init file for the dry_torch package."""
import warnings

from dry_torch.checkpoint import CheckpointIO
from dry_torch.checkpoint import ModelStateIO
from dry_torch.evaluating import Diagnostic
from dry_torch.evaluating import Test
from dry_torch.evaluating import Validation
from dry_torch.exceptions import FailedOptionalImportWarning
from dry_torch.experiments import ChildExperiment
from dry_torch.experiments import Experiment
from dry_torch.experiments import ParentExperiment
from dry_torch.metrics import Loss
from dry_torch.metrics import Metric
from dry_torch.loading import DataLoader
from dry_torch.learning import LearningScheme
from dry_torch.learning import Model
from dry_torch.tracking import remove_all_default_trackers
from dry_torch.tracking import extend_default_trackers
from dry_torch.tracking import Tracker
from dry_torch.training import Trainer

from dry_torch.trackers import logging


def add_standard_trackers_to_default_trackers() -> None:
    """Add a list of trackers to the default ones."""
    tracker_list: list[Tracker] = [
        builtin_logger.BuiltinLogger(),
    ]
    try:
        import tqdm

    except (ImportError, ModuleNotFoundError) as ie:
        warnings.warn(FailedOptionalImportWarning('tqdm', ie))

    else:
        from dry_torch.trackers import tqdm
        tracker_list.append(tqdm_logger.TqdmLogger())

    try:
        import yaml  # type: ignore
    except (ImportError, ModuleNotFoundError) as ie:
        warnings.warn(FailedOptionalImportWarning('yaml', ie))
    else:
        from dry_torch.trackers import yaml
        tracker_list.append(yaml_dumper.YamlDumper())
    extend_default_trackers(tracker_list)


add_standard_trackers_to_default_trackers()

"""Init file for the dry_torch package."""

__all__ = [
    'Loss',
    'Metric',
    'ModelStateIO',
    'CheckpointIO',
    'DataLoader',
    'LearningScheme',
    'Model',
    'Diagnostic',
    'Validation',
    'Test',
    'Experiment',
    'Trainer',
]
from src.dry_torch.calculating import Loss
from src.dry_torch.calculating import Metric
from src.dry_torch.checkpoint import CheckpointIO
from src.dry_torch.checkpoint import ModelStateIO
from src.dry_torch.loading import DataLoader
from src.dry_torch.learning import LearningScheme
from src.dry_torch.learning import Model
from src.dry_torch.evaluating import Diagnostic
from src.dry_torch.evaluating import Test
from src.dry_torch.evaluating import Validation
from src.dry_torch.tracking import DEFAULT_TRACKERS
from src.dry_torch.tracking import Experiment
from src.dry_torch.tracking import Tracker
from src.dry_torch.training import Trainer

from src.dry_torch.trackers import builtin_logger


def extend_default_trackers(tracker_list: list[Tracker]) -> None:
    """Add a list of trackers to the default ones."""
    for tracker in tracker_list:
        DEFAULT_TRACKERS[tracker.__class__.__name__] = tracker


def remove_all_default_trackers() -> None:
    """Remove all default trackers."""
    DEFAULT_TRACKERS.clear()


def add_standard_trackers_to_default_trackers() -> None:
    """Add a list of trackers to the default ones."""
    tracker_list: list[Tracker] = [
        builtin_logger.BuiltinLogger(),
    ]
    try:
        import tqdm

    except ImportError:
        pass

    else:
        from src.dry_torch.trackers import tqdm_logger
        tracker_list.append(tqdm_logger.TqdmLogger())

    try:
        import yaml  # type: ignore
    except ImportError:
        pass
    else:
        from src.dry_torch.trackers import yaml_dumper
        tracker_list.append(yaml_dumper.YamlDumper())
    extend_default_trackers(tracker_list)


add_standard_trackers_to_default_trackers()

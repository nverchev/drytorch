"""Init file for dry_torch module."""

__all__ = [
    'ModelStateIO',
    'CheckpointIO',
    'Experiment',
    'DataLoader',
    'SimpleLossCalculator',
    'MetricsCalculator',
    'LearningScheme',
    'Model',
    'Trainer',
    'Diagnostic',
    'Validation',
    'Test',
    'Plotter',
    'register_model',
]

from src.dry_torch.plotting import Plotter
from src.dry_torch.checkpoint import ModelStateIO
from src.dry_torch.checkpoint import CheckpointIO
from src.dry_torch.loading import DataLoader
from src.dry_torch.calculating import SimpleLossCalculator
from src.dry_torch.calculating import MetricsCalculator
from src.dry_torch.learning import LearningScheme
from src.dry_torch.learning import Model
from src.dry_torch.training import Trainer
from src.dry_torch.evaluating import Diagnostic
from src.dry_torch.evaluating import Validation
from src.dry_torch.evaluating import Test
from src.dry_torch.registering import register_model
from src.dry_torch.tracking import DEFAULT_TRACKERS
from src.dry_torch.tracking import Experiment
from src.dry_torch.tracking import Tracker

from src.dry_torch.trackers import builtin_logger
from src.dry_torch.trackers import metadata


def extend_default_trackers(tracker_list: list[Tracker]) -> None:
    for tracker in tracker_list:
        DEFAULT_TRACKERS[tracker.__class__.__name__] = tracker


def remove_all_default_trackers() -> None:
    DEFAULT_TRACKERS.clear()


def add_default_trackers() -> None:
    tracker_list: list[Tracker] = [
        metadata.MetadataExtractor(),
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


add_default_trackers()

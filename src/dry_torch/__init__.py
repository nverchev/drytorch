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

from src.dry_torch.trackers import builtin_logger
from src.dry_torch.trackers import metadata

DEFAULT_TRACKERS.extend([
    metadata.MetadataExtractor(),
    builtin_logger.BuiltinLogger(),
])

try:
    import tqdm
except ImportError:
    TQDM_FLAG = False

else:
    from src.dry_torch.trackers import tqdm_logger

    TQDM_FLAG = True
    tqdm_tracker = tqdm_logger.TqdmLogger()
    DEFAULT_TRACKERS.append(tqdm_tracker)

try:
    import yaml  # type: ignore
except ImportError:
    pass
else:
    from src.dry_torch.trackers import yaml_dumper

    DEFAULT_TRACKERS.append(yaml_dumper.YamlDumper())


def remove_all_default_trackers():
    DEFAULT_TRACKERS.clear()


def set_compact_mode():
    if TQDM_FLAG:
        builtin_logger.set_verbosity(builtin_logger.INFO_LEVELS.training)
        tqdm_tracker.enable_training_bar = True
    else:
        builtin_logger.set_verbosity(builtin_logger.INFO_LEVELS.epoch)

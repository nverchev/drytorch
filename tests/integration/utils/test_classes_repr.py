"""Test the automatic representation for the library classes."""

from collections.abc import Generator

import pytest

from drytorch.lib.gradient_ops import HistClipper
from drytorch.lib.hooks import EarlyStoppingCallback, HookRegistry
from drytorch.lib.schedulers import ExponentialScheduler, WarmupScheduler
from drytorch.utils.average import get_trailing_mean
from drytorch.utils.repr_utils import recursive_repr


@pytest.fixture(autouse=True, scope='module')
def start_experiment(run) -> Generator[None, None, None]:
    """Create an experimental scope for the tests."""
    yield
    return


def test_repr_trainer(identity_trainer, mocker) -> None:
    """Test Trainer, Model, DataLoader, and objective basic repr."""
    expected = {
        'class': 'Trainer',
        'learning_schema': {
            'class': 'LearningSchema',
            'base_lr': 0.1,
            'optimizer_cls': 'Adam',
            'optimizer_defaults': {'betas': (0.9, 0.999)},
            'scheduler': 'ConstantScheduler()',
        },
        'loader': {
            'class': 'DataLoader',
            'batch_size': 4,
            'dataset': {'class': 'IdentityDataset', 'len_epoch': 64},
            'dataset_len': 64,
        },
        'model': {
            'class': 'Model',
            'checkpoint': 'LocalCheckpoint',
            'epoch': 0,
            'mixed_precision': False,
            'module': {'class': 'Linear', 'training': True},
        },
        'objective': {
            'class': 'Loss',
            'criterion': "operator.itemgetter('MSE')",
            'formula': '[MSE]',
            'higher_is_better': False,
            'name': 'MSE',
            'named_fn': {'MSE': 'mse'},
        },
        'post_epoch_hooks': 'HookRegistry',
        'pre_epoch_hooks': 'HookRegistry',
    }

    assert recursive_repr(identity_trainer) == expected


def test_hook_repr() -> None:
    """Test the representation of a hook registry."""
    registry = HookRegistry()
    registry.register(EarlyStoppingCallback(filter_fn=get_trailing_mean(9)))
    expected = {
        'class': 'HookRegistry',
        'hooks': [
            {
                'class': 'EarlyStoppingCallback',
                'monitor': {
                    'class': 'MetricMonitor',
                    'extractor': 'MetricExtractor',
                    'metric_tracker': {
                        'best_is': 'auto',
                        'class': 'MetricTracker',
                        'filter_fn': 'trailing_mean(window_size=9)',
                        'min_delta': 1e-08,
                        'patience': 10,
                    },
                },
                'start_from_epoch': 2,
            }
        ],
    }
    assert recursive_repr(registry) == expected


def test_gradient_op_repr() -> None:
    """Test the representation of a gradient op."""
    expected = {
        'class': 'HistClipper',
        'criterion': {
            'alpha': 0.97,
            'class': 'ZStatCriterion',
            'clipping_function': 'reciprocal_clipping',
            'z_thresh': 2.5,
        },
        'n_warmup_steps': 20,
        'warmup_clip_strategy': {'class': 'GradNormClipper', 'threshold': 1.0},
    }

    assert recursive_repr(HistClipper()) == expected


def test_scheduler_repr() -> None:
    """Test the representation of a scheduler."""
    expected = {
        'class': 'WarmupScheduler',
        'base_scheduler': {
            'class': 'ExponentialScheduler',
            'exp_decay': 0.975,
            'min_decay': 0.0,
        },
        'warmup_steps': 2,
    }
    scheduler = WarmupScheduler(ExponentialScheduler(), 2)
    assert recursive_repr(scheduler) == expected

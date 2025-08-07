"""Test the automatic representation for the library classes."""

from collections.abc import Generator

import pytest

from drytorch.gradient_ops import HistClipping
from drytorch.hooks import EarlyStoppingCallback, HookRegistry
from drytorch.schedulers import ExponentialScheduler, WarmupScheduler
from drytorch.utils.repr_utils import recursive_repr
from drytorch.utils.statistics import get_moving_average


@pytest.fixture(autouse=True, scope='module')
def start_experiment(experiment) -> Generator[None, None, None]:
    """Create an experimental scope for the tests."""
    yield
    return


def test_repr_trainer(identity_trainer):
    """Test Trainer, Model, DataLoader, and objective basic repr."""
    expected = {'class': 'Trainer',
                'learning_scheme': {'class': 'LearningScheme',
                                    'base_lr': 0.1,
                                    'optimizer_cls': 'Adam',
                                    'optimizer_defaults': {
                                        'betas': (0.9, 0.999)
                                    },
                                    'scheduler': 'ConstantScheduler()'},
                'loader': {'class': 'DataLoader',
                           'batch_size': 4,
                           'dataset': {'class': 'IdentityDataset',
                                       'len_epoch': 64},
                           'dataset_len': 64},
                'model': {'class': 'Model',
                          'checkpoint': 'LocalCheckpoint',
                          'epoch': 0,
                          'mixed_precision': False,
                          'module': {'class': 'Linear', 'training': True}},
                'objective': {'class': 'Loss',
                              'criterion': "operator.itemgetter('MSE')",
                              'formula': '[MSE]',
                              'fun': 'mse',
                              'higher_is_better': False,
                              'name': 'MSE',
                              'named_metric_fun': {'MSE': 'mse'}},
                'post_epoch_hooks': 'HookRegistry',
                'pre_epoch_hooks': 'HookRegistry'}

    assert recursive_repr(identity_trainer) == expected


def test_hook_repr():
    """Test the representation of a hook registry."""
    registry = HookRegistry()
    registry.register(
        EarlyStoppingCallback(filter_fn=get_moving_average(.8))
    )
    expected = {
        'class': 'HookRegistry',
        'hooks': [
            {
                'class': 'EarlyStoppingCallback',
                'monitor': {
                    'filter_fn':
                        'moving_average(decay=0.8, mass_coverage=0.99)',
                    'best_is': 'auto',
                    'class': 'MetricMonitor',
                    'min_delta': 1e-08,
                    'patience': 10,
                },
                'start_from_epoch': 2,
            },
        ],
    }
    assert recursive_repr(registry) == expected


def test_gradient_op_repr():
    """Test the representation of a gradient op."""
    expected = {'class': 'HistClipping',
                'criterion': {'alpha': 0.97,
                              'class': 'ZStatCriterion',
                              'clipping_function': 'reciprocal_clipping',
                              'z_thresh': 2.5},
                'n_warmup_steps': 20,
                'warmup_clip_strategy': {'class': 'GradNormClipper',
                                         'threshold': 1.0}}

    assert recursive_repr(HistClipping()) == expected


def test_scheduler_repr():
    """Test the representation of a scheduler."""
    expected = {'class': 'WarmupScheduler',
                'base_scheduler': {'class': 'ExponentialScheduler',
                                   'exp_decay': 0.975,
                                   'min_decay': 0.0},
                'warmup_steps': 2}
    scheduler = WarmupScheduler(ExponentialScheduler(), 2)
    assert recursive_repr(scheduler) == expected

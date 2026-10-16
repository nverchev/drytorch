"""Test the automatic representation for the library classes."""

from collections.abc import Generator

import pytest

from drytorch.lib.evaluations import Validation
from drytorch.lib.gradient_ops import HistClipper
from drytorch.lib.hooks import EarlyStopping, HookChain
from drytorch.lib.monitoring import Watch
from drytorch.lib.schedulers import ExponentialScheduler, warmup
from drytorch.utils.averages import get_trailing_mean
from drytorch.utils.repr_utils import recursive_repr


@pytest.fixture(autouse=True, scope='module')
def start_experiment(run) -> Generator[None, None, None]:
    """Create an experimental scope for the tests."""
    yield
    return


def test_repr_trainer(identity_trainer) -> None:
    """Test Trainer, Model, DataLoader, and objective basic repr."""
    expected = {
        'class': 'Trainer',
        'learning_schema': {
            'class': 'LearningSchema',
            'gradient_op': 'NoOp',
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
            'sampler': {
                'class': 'RandomSampler',
                'data_source': 'range(0, 64)',
                'replacement': False,
            },
        },
        'model': {
            'class': 'Model',
            'checkpoint': 'LocalCheckpoint',
            'exec_module': {'class': 'Linear', 'training': True},
        },
        'objective': {
            'class': 'Loss',
            'criterion': "operator.itemgetter('MSE')",
            'formula': '[MSE]',
            'higher_is_better': False,
            'name': 'MSE',
            'named_fn': {'MSE': 'mse'},
        },
        'post_epoch_hooks': 'HookChain',
        'pre_epoch_hooks': 'HookChain',
    }

    identity_trainer.model.exec_module = identity_trainer.model.module

    assert recursive_repr(identity_trainer) == expected


def test_shared_model_is_represented_once(
    identity_trainer, identity_loader, square_loss_calc
) -> None:
    """Test a model used twice is represented once."""
    validation = Validation(
        identity_trainer.model,
        name='SharedValidation',
        loader=identity_loader,
        metric=square_loss_calc,
    )

    recorded = recursive_repr(
        {'trainer': identity_trainer, 'validation': validation}
    )

    assert recorded['validation']['model'] is recorded['trainer']['model']


def test_hook_repr() -> None:
    """Test the representation of a hook and of the condition it runs on."""
    chain = HookChain(
        [EarlyStopping(watch=Watch(filter_fn=get_trailing_mean(9)))]
    )
    expected = {
        'class': 'HookChain',
        'hooks': (
            {
                'class': 'EarlyStopping',
                'when': {
                    'class': 'AndCondition',
                    'left': {'class': 'After', 'epoch': 2},
                    'right': {
                        'class': 'Plateau',
                        'min_delta': 1e-08,
                        'patience': 10,
                        'watch': {
                            'class': 'Watch',
                            'filter_fn': 'trailing_mean(window_size=9)',
                        },
                    },
                },
            },
        ),
    }

    assert recursive_repr(chain) == expected


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
        'scheduler': {
            'class': 'ExponentialScheduler',
            'exp_decay': 0.975,
            'min_decay': 0.0,
        },
        'warmup_steps': 2,
    }
    scheduler = ExponentialScheduler().compose(warmup(2))

    assert recursive_repr(scheduler) == expected

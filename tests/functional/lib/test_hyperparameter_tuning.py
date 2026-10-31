"""Functional tests for simple hyperparameter tuning."""

import dataclasses
import gc

from collections.abc import Generator, MutableMapping

import pytest

from drytorch import Trainer
from drytorch.core import registering
from drytorch.core.experimenting import Run
from drytorch.lib import hooks
from drytorch.lib.models import Model
from drytorch.utils import repr_utils
from tests.simple_classes import Linear


# pytorch dynamo deprecation warnings for torch==2.10 under Python 3.14+
pytestmark = pytest.mark.filterwarnings(
    'ignore:.*_UnionGenericAlias.*:DeprecationWarning'
)


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


@pytest.fixture
def benchmark_values() -> MutableMapping[int, float | None]:
    """Thresholds for the first epochs with None values."""
    return {}.fromkeys(range(1, 5))


def test_automatic_names(
    mocker,
    standard_learning_schema,
    square_loss_calc,
    linear_model,
    identity_loader,
) -> None:
    """Test the creation of models in a loop with automatic names."""
    # models created by earlier tests advance the shared name counters
    mocker.patch.object(Model._name, '_counter', repr_utils.NameCounter())
    results: set[str] = set()
    module = linear_model.module
    registering.unregister_model(linear_model)
    for lr_pow in range(4):
        training_loder, val_loader = identity_loader.split()
        lr = 10 ** (-lr_pow)
        linear_model_copy = Model(module)
        new_learning_schema = dataclasses.replace(
            standard_learning_schema, base_lr=lr
        )
        builder = Trainer.builder(
            linear_model_copy,
            name='MyTrainer',
            loader=training_loder,
            learning_schema=new_learning_schema,
            loss=square_loss_calc.copy(),
        )
        builder.add_validation(val_loader=val_loader)
        builder.add_early_stopping()
        trainer = builder.build()
        trainer.train(10)
        results.add(linear_model_copy.name)
        registering.unregister_model(linear_model_copy)
        gc.collect()

    assert results == {'Model', 'Model_1', 'Model_2', 'Model_3'}


def test_iterative_pruning(
    benchmark_values,
    standard_learning_schema,
    square_loss_calc,
    identity_loader,
) -> None:
    """Test a pruning strategy that requires model improvement at each epoch."""
    first_trial_len = 0
    for lr in (0.001, 0.01, 0.1, 10):  # the last learning rate diverges
        training_loder, val_loader = identity_loader.split()
        trial_model = Model(Linear(1, 1))  # every trial starts from zero
        new_learning_schema = dataclasses.replace(
            standard_learning_schema, base_lr=lr
        )
        builder = Trainer.builder(
            trial_model,
            name='MyTrainer',
            loader=training_loder,
            learning_schema=new_learning_schema,
            loss=square_loss_calc.copy(),
        )
        builder.add_validation(val_loader=val_loader)
        watch = builder.watch(best_is='lower')
        prune_callback = hooks.Pruning(benchmark_values, watch=watch)
        builder.add_post_epoch_hook(prune_callback)
        trainer = builder.build()
        trainer.train(4)
        if lr == 0.001:  # first trial has no thresholds
            first_trial_len = len(prune_callback.when.monitor.values)

        benchmark_values = prune_callback.when.monitor.values
        registering.unregister_model(trial_model)
        gc.collect()

    assert first_trial_len == 4
    # the last run (worst LR) should be immediately pruned at epoch 1.
    assert len(benchmark_values) == 1

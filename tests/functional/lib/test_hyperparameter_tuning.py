"""Functional tests for simple hyperparameter tuning."""

import dataclasses
import gc

from collections.abc import Generator, MutableMapping
from typing import Any

import pytest

from drytorch import Trainer
from drytorch.core import registering
from drytorch.core.experimenting import Run
from drytorch.lib import hooks
from drytorch.lib.models import Model


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
    standard_learning_schema, square_loss_calc, linear_model, identity_loader
) -> None:
    """Test the creation of models in a loop with automatic names."""
    results = dict[str, float]()
    module = linear_model.module
    registering.unregister_model(linear_model)
    for lr_pow in range(4):
        training_loder, val_loader = identity_loader.split()
        lr = 10 ** (-lr_pow)
        linear_model_copy = Model(module)
        new_learning_schema = dataclasses.replace(
            standard_learning_schema, base_lr=lr
        )

        trainer = Trainer(
            linear_model_copy,
            name='MyTrainer',
            loader=training_loder,
            learning_schema=new_learning_schema,
            loss=square_loss_calc,
        )
        trainer.add_validation(val_loader)
        early_stopping = hooks.EarlyStoppingCallback[Any, Any]()
        trainer.post_epoch_hooks.register(early_stopping)
        trainer.train(10)
        results[linear_model_copy.name] = early_stopping.monitor.best_value
        registering.unregister_model(linear_model_copy)
        gc.collect()

    assert {'Model', 'Model_1', 'Model_2', 'Model_3'} == set(results)


def test_iterative_pruning(
    benchmark_values,
    standard_learning_schema,
    linear_model,
    square_loss_calc,
    identity_loader,
) -> None:
    """Test a pruning strategy that requires model improvement at each epoch."""
    registering.unregister_model(linear_model)
    first_trial_len = 0
    for lr_pow in reversed(range(4)):
        training_loder, val_loader = identity_loader.split()
        lr = 10 ** (-lr_pow)
        linear_model_copy = Model(linear_model.module)
        new_learning_schema = dataclasses.replace(
            standard_learning_schema, base_lr=lr
        )
        trainer = Trainer(
            model=linear_model_copy,
            name='MyTrainer',
            loader=training_loder,
            learning_schema=new_learning_schema,
            loss=square_loss_calc,
        )
        trainer.add_validation(val_loader)
        prune_callback = hooks.PruneCallback[Any, Any](
            benchmark_values, best_is='lower'
        )
        trainer.post_epoch_hooks.register(prune_callback)
        trainer.train(4)
        if lr_pow == 3:  # First trial (best LR) establishes curve
            first_trial_len = len(prune_callback.trial_values)

        benchmark_values = prune_callback.trial_values
        registering.unregister_model(linear_model_copy)
        gc.collect()

    assert first_trial_len == 4
    # the last run (worst LR) should be immediately pruned at epoch 1.
    assert len(benchmark_values) == 1

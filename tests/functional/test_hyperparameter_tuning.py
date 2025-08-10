"""Functional tests for simple hyperparameter tuning."""

from collections.abc import Generator, MutableMapping

import pytest

from drytorch import Model, Trainer, hooks
from drytorch.core.experiments import Run
from tests.simple_classes import Linear


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


@pytest.fixture
def benchmark_values() -> MutableMapping[int, float | None]:
    """Thresholds for the first epochs with None values."""
    return {}.fromkeys(range(1, 5))


def test_automatic_names(standard_learning_scheme,
                         square_loss_calc,
                         identity_loader) -> None:
    """Test the creation of models in a loop with automatic names."""
    results = dict[str, float]()
    for lr_pow in range(4):
        training_loder, val_loader = identity_loader.split()
        linear_model = Model(Linear(1, 1))
        lr = 10 ** (-lr_pow)
        standard_learning_scheme.base_lr = lr
        trainer = Trainer(linear_model,
                          name='MyTrainer',
                          loader=training_loder,
                          learning_scheme=standard_learning_scheme,
                          loss=square_loss_calc)
        trainer.add_validation(val_loader)
        early_stopping = hooks.EarlyStoppingCallback()
        trainer.post_epoch_hooks.register(early_stopping)
        trainer.train(10)
        results[linear_model.name] = early_stopping.monitor.best_value

    assert {'Model', 'Model_1', 'Model_2', 'Model_3'} == set(results)


def test_iterative_pruning(benchmark_values,
                           standard_learning_scheme,
                           square_loss_calc,
                           identity_loader) -> None:
    """Test a pruning strategy that requires model improvement at each epoch."""
    for lr_pow in range(4):
        training_loder, val_loader = identity_loader.split()
        linear_model = Model(Linear(1, 1))
        lr = 10 ** (-lr_pow)
        standard_learning_scheme.base_lr = lr
        trainer = Trainer(linear_model,
                          name='MyTrainer',
                          loader=training_loder,
                          learning_scheme=standard_learning_scheme,
                          loss=square_loss_calc)
        trainer.add_validation(val_loader)
        prune_callback = hooks.PruneCallback(benchmark_values, best_is='lower')
        trainer.post_epoch_hooks.register(prune_callback)
        trainer.train(4)
        benchmark_values = prune_callback.trial_values

    # the last run should be immediately pruned.
    assert len(benchmark_values) <= 1

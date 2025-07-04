"""Module that tests training"""
from operator import itemgetter

import pytest

from typing import Generator

import torch

import drytorch
from drytorch import schedulers, Experiment, Model, Trainer

from drytorch import hooks

from simple_classes import Linear


@pytest.fixture(autouse=True, scope='module')
def experiment(tmpdir_factory) -> Generator[Experiment, None, None]:
    """Fixture of an experiment."""
    drytorch.remove_all_default_trackers()
    par_dir = tmpdir_factory.mktemp('experiments')
    with Experiment[None](name='TestExperiment', par_dir=par_dir) as exp:
        yield exp
    return


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
        results[linear_model.name] = early_stopping.monitor.best_result

    assert {'Model', 'Model_1', 'Model_2', 'Model_3'} == set(results)


def test_iterative_pruning(standard_learning_scheme,
                           square_loss_calc,
                           identity_loader) -> None:
    """Test a pruning strategy that requires model to improve at each epoch."""
    benchmark_values = {1: 1., 2.: 1, 3.: 1, 4: 1}
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

    # last run should be immediately pruned.
    assert not benchmark_values

"""Functional tests for the Trainer class and some hooks."""

from collections.abc import Generator

import torch

import pytest

from drytorch.core.experiment import Run
from drytorch.lib import hooks, schedulers


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


def test_convergence(identity_trainer) -> None:
    """Test trainer convergence to 1."""
    identity_trainer.train(10)
    linear_weight = next(identity_trainer.model.module.parameters())
    assert torch.isclose(linear_weight, torch.tensor(1.0), atol=0.1)


def test_early_stopping(square_loss_calc, identity_trainer) -> None:
    """Test early stopping when monitoring training."""
    hook = hooks.EarlyStoppingCallback(
        square_loss_calc, patience=2, min_delta=1
    )
    identity_trainer.post_epoch_hooks.register(hook)
    identity_trainer.train(4)
    # 2 epochs of patience and terminate at 3 or 4
    assert identity_trainer.model.epoch in {3, 4}


def test_early_stopping_on_val(
    identity_loader, square_loss_calc, identity_trainer
) -> None:
    """Test early stopping when monitoring validation."""
    hook = hooks.EarlyStoppingCallback(
        square_loss_calc, patience=2, min_delta=1
    )
    identity_trainer.add_validation(val_loader=identity_loader)
    identity_trainer.post_epoch_hooks.register(hook)
    identity_trainer.train(4)
    # 5 epochs of patience and terminate at 3
    assert identity_trainer.model.epoch == 3


def test_pruning_callback(square_loss_calc, identity_trainer) -> None:
    """Test pruning based on metric thresholds."""
    pruning_thresholds = {2: 1.0, 3: 0.0}
    identity_trainer.post_epoch_hooks.register(
        hooks.PruneCallback(
            thresholds=pruning_thresholds,
            metric=square_loss_calc,
        )
    )
    identity_trainer.train(4)
    # stop at epoch 3 because the loss is always greater than 0
    assert identity_trainer.model.epoch == 3


def test_reduce_lr_on_plateau(square_loss_calc, identity_trainer) -> None:
    """Test learning rate reduction on plateau."""
    factor = 0.1
    initial_lr = identity_trainer._model_optimizer.base_lr
    identity_trainer.post_epoch_hooks.register(
        hooks.ReduceLROnPlateau(
            metric=square_loss_calc,
            factor=factor,
            min_delta=0.1,
        )
    )
    identity_trainer.train(5)
    final_lr = identity_trainer._model_optimizer.get_scheduled_lr(initial_lr)
    assert final_lr == pytest.approx(factor * initial_lr)


def test_restart_schedule_on_plateau(
    square_loss_calc, identity_trainer
) -> None:
    """Test learning rate schedule restart on plateau."""
    exp_scheduler = schedulers.ExponentialScheduler()
    initial_lr = identity_trainer._model_optimizer.base_lr
    identity_trainer.update_learning_rate(scheduler=exp_scheduler)
    identity_trainer.post_epoch_hooks.register(
        hooks.RestartScheduleOnPlateau(metric=square_loss_calc)
    )
    identity_trainer.train(5)
    # training should complete with schedule restarts
    final_lr = identity_trainer._model_optimizer.get_scheduled_lr(initial_lr)
    assert not final_lr == pytest.approx(exp_scheduler(initial_lr, 4))

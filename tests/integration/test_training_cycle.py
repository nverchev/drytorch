"""Module that tests training"""

import torch

from src.dry_torch import schedulers
from src.dry_torch import hooks


def test_convergence(identity_trainer) -> None:
    """ Trainer works if the model weight converges to 1."""
    identity_trainer.train(6)
    linear_weight = next(identity_trainer.model.module.parameters())
    assert torch.isclose(linear_weight, torch.tensor(1.))


def test_early_stopping(identity_loader,
                        linear_model,
                        square_loss_calc,
                        zero_metrics_calc,
                        identity_trainer) -> None:
    """Test early stopping when monitoring training."""
    hook = hooks.EarlyStoppingCallback(square_loss_calc,
                                       patience=5,
                                       min_delta=10)
    identity_trainer._post_epoch_hooks.register(hook)
    identity_trainer.train(15)
    # patience never reset, 5 epochs of patience and terminate at 6
    assert identity_trainer.model.epoch == 6


def test_early_stopping_on_val(identity_loader,
                               linear_model,
                               square_loss_calc,
                               identity_trainer) -> None:
    """Test early stopping when monitoring validation."""
    hook = hooks.EarlyStoppingCallback(square_loss_calc,
                                       patience=5,
                                       min_delta=10)
    identity_trainer.add_validation(val_loader=identity_loader)
    identity_trainer._post_epoch_hooks.register(hook)
    identity_trainer.train(15)
    assert identity_trainer.model.epoch == 6


def test_pruning_callback(identity_loader,
                          linear_model,
                          square_loss_calc,
                          identity_trainer) -> None:
    """Test pruning based on metric thresholds."""
    pruning_thresholds = {2: 0.1, 4: 0}
    identity_trainer._post_epoch_hooks.register(
        hooks.PruneCallback(
            pruning=pruning_thresholds,
            metric=square_loss_calc,
        )
    )
    identity_trainer.train(6)
    # Should stop at epoch 4 when loss goes below 0.05
    assert identity_trainer.model.epoch == 4


def test_reduce_lr_on_plateau(identity_loader,
                              linear_model,
                              square_loss_calc,
                              identity_trainer) -> None:
    """Test learning rate reduction on plateau."""
    factor = 0.01
    initial_lr = identity_trainer._model_optimizer.base_lr
    identity_trainer._post_epoch_hooks.register(
        hooks.ReduceLROnPlateau(
            metric=square_loss_calc,
            factor=factor,
        )
    )
    identity_trainer.train(8)
    final_lr = identity_trainer._model_optimizer.get_scheduled_lr(initial_lr)
    # LR should have been reduced at least once
    assert final_lr == factor * initial_lr


def test_restart_schedule_on_plateau(identity_loader,
                                     linear_model,
                                     square_loss_calc,
                                     identity_trainer) -> None:
    """Test learning rate schedule restart on plateau."""
    exp_scheduler = schedulers.ExponentialScheduler()
    initial_lr = identity_trainer._model_optimizer.base_lr
    identity_trainer.update_learning_rate(scheduler=exp_scheduler)
    identity_trainer._post_epoch_hooks.register(
        hooks.RestartScheduleOnPlateau(
            metric=square_loss_calc,
            cooldown=1
        )
    )
    identity_trainer.train(8)
    # Training should complete with schedule restarts
    final_lr = identity_trainer._model_optimizer.get_scheduled_lr(initial_lr)
    assert final_lr > exp_scheduler(initial_lr, 8)


def test_multiple_callbacks(identity_loader,
                            linear_model,
                            square_loss_calc,
                            identity_trainer) -> None:
    """Test interaction between multiple callbacks."""
    identity_trainer.add_validation(val_loader=identity_loader)

    # Register multiple callbacks
    identity_trainer._post_epoch_hooks.register_all([
        hooks.EarlyStoppingCallback(patience=2),
        hooks.ReduceLROnPlateau(
            metric=square_loss_calc,
            patience=1
        ),
        hooks.PruneCallback(
            pruning={5: 0.01},
            metric=square_loss_calc
        )
    ])

    identity_trainer.train(10)
    # Should stop before max epochs due to either early stopping or pruning
    assert identity_trainer.model.epoch < 10

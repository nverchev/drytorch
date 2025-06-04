"""Module that tests training"""

import torch

from dry_torch import schedulers
from dry_torch import hooks


def test_convergence(identity_trainer) -> None:
    """ Trainer works if the model weight converges to 1."""
    identity_trainer.train(4)
    linear_weight = next(identity_trainer.model.module.parameters())
    assert torch.isclose(linear_weight, torch.tensor(1.), atol=0.1)


def test_early_stopping(identity_loader,
                        linear_model,
                        square_loss_calc,
                        zero_metrics_calc,
                        identity_trainer) -> None:
    """Test early stopping when monitoring training."""
    hook = hooks.EarlyStoppingCallback(square_loss_calc,
                                       patience=2,
                                       min_delta=1)
    identity_trainer.post_epoch_hooks.register(hook)
    identity_trainer.train(4)
    # 5 epochs of patience and terminate at 3
    assert identity_trainer.model.epoch == 3


def test_early_stopping_on_val(identity_loader,
                               linear_model,
                               square_loss_calc,
                               identity_trainer) -> None:
    """Test early stopping when monitoring validation."""
    hook = hooks.EarlyStoppingCallback(square_loss_calc,
                                       patience=2,
                                       min_delta=1)
    identity_trainer.add_validation(val_loader=identity_loader)
    identity_trainer.post_epoch_hooks.register(hook)
    identity_trainer.train(4)
    # 5 epochs of patience and terminate at 3
    assert identity_trainer.model.epoch == 3


def test_pruning_callback(identity_loader,
                          linear_model,
                          square_loss_calc,
                          identity_trainer) -> None:
    """Test pruning based on metric thresholds."""
    pruning_thresholds = {2: 1., 3: 0.}
    identity_trainer.post_epoch_hooks.register(
        hooks.PruneCallback(
            pruning=pruning_thresholds,
            metric=square_loss_calc,
        )
    )
    identity_trainer.train(4)
    # stop at epoch 3 because loss is always greater than 0
    assert identity_trainer.model.epoch == 3


def test_reduce_lr_on_plateau(identity_loader,
                              linear_model,
                              square_loss_calc,
                              identity_trainer) -> None:
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
    identity_trainer.train(2)
    final_lr = identity_trainer._model_optimizer.get_scheduled_lr(initial_lr)
    assert final_lr == factor * initial_lr


def test_restart_schedule_on_plateau(identity_loader,
                                     linear_model,
                                     square_loss_calc,
                                     identity_trainer) -> None:
    """Test learning rate schedule restart on plateau."""
    exp_scheduler = schedulers.ExponentialScheduler()
    initial_lr = identity_trainer._model_optimizer.base_lr
    identity_trainer.update_learning_rate(scheduler=exp_scheduler)
    identity_trainer.post_epoch_hooks.register(
        hooks.RestartScheduleOnPlateau(
            metric=square_loss_calc,
            cooldown=1
        )
    )
    identity_trainer.train(4)
    # Training should complete with schedule restarts
    final_lr = identity_trainer._model_optimizer.get_scheduled_lr(initial_lr)
    assert final_lr > exp_scheduler(initial_lr, 4)

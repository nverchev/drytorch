"""Tests for the scheduler module."""

import numpy as np
from src.dry_torch.schedulers import ConstantScheduler
from src.dry_torch.schedulers import ExponentialScheduler, CosineScheduler


def test_constant_scheduler() -> None:
    """Test ConstantScheduler scheduling."""
    scheduler = ConstantScheduler()
    base_lr = 0.1
    epochs = [0, 10, 50]

    for epoch in epochs:
        assert scheduler(base_lr, epoch) == base_lr


def test_exponential_scheduler() -> None:
    """Test ExponentialScheduler scheduling."""
    exp_decay = 0.9
    min_decay = 0.5
    scheduler = ExponentialScheduler(exp_decay=exp_decay, min_decay=min_decay)

    base_lr = 1.0
    epoch_0 = scheduler(base_lr, 0)  # Epoch 0 should return base_lr
    epoch_1 = scheduler(base_lr, 1)  # Epoch 1 should return base_lr * exp_decay
    epoch_10 = scheduler(base_lr, 10)  # Exponentially decayed value

    assert epoch_0 == base_lr
    assert epoch_1 == base_lr * exp_decay
    assert epoch_10 == min_decay


def test_cosine_scheduler() -> None:
    """Test CosineScheduler scheduling."""
    decay_steps = 100
    min_decay = 0.1
    scheduler = CosineScheduler(decay_steps=decay_steps, min_decay=min_decay)
    base_lr = 1.0

    # Start of the schedule
    lr_epoch_0 = scheduler(base_lr, 0)
    assert lr_epoch_0 == base_lr

    # Midway through the schedule
    lr_epoch_mid = scheduler(base_lr, decay_steps // 2)
    expected_lr_mid = (
        min_decay * base_lr +
        (base_lr - min_decay * base_lr) * (1 + np.cos(np.pi / 2)) / 2
    )
    assert np.isclose(lr_epoch_mid, expected_lr_mid, rtol=1e-5)

    # End of the schedule
    lr_epoch_end = scheduler(base_lr, decay_steps)
    assert lr_epoch_end == min_decay * base_lr

    # Beyond decay_steps, learning rate should remain constant at min_lr
    lr_beyond_decay = scheduler(base_lr, decay_steps + 10)
    assert lr_beyond_decay == min_decay * base_lr

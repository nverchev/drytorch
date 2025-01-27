"""Tests for the scheduler module."""

import pytest

import numpy as np

from src.dry_torch.schedulers import ExponentialScheduler, CosineScheduler
from src.dry_torch.schedulers import ConstantScheduler


class TestConstantScheduler:
    """Test ConstantScheduler functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the ConstantScheduler for testing."""
        self.scheduler = ConstantScheduler()

    def test_constant_scheduler(self) -> None:
        """Test that ConstantScheduler returns the same learning rate."""
        base_lr = 0.1
        epochs = [0, 10, 50]

        for epoch in epochs:
            assert self.scheduler(base_lr, epoch) == base_lr


# Test class for ExponentialScheduler
class TestExponentialScheduler:
    """Test ExponentialScheduler functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the ExponentialScheduler for testing."""
        self.exp_decay = 0.9
        self.min_decay = 0.5
        self.scheduler = ExponentialScheduler(exp_decay=self.exp_decay,
                                              min_decay=self.min_decay)

    def test_exponential_scheduler(self) -> None:
        """Test that ExponentialScheduler correctly decays learning rate."""
        base_lr = 1.0

        # Epoch 0 should return base_lr
        assert self.scheduler(base_lr, 0) == base_lr

        # Epoch 1 should return base_lr * exp_decay
        assert self.scheduler(base_lr, 1) == base_lr * self.exp_decay

        # Beyond decay limit, it should return min_decay * base_lr
        assert self.scheduler(base_lr, 10) == self.min_decay


# Test class for CosineScheduler
class TestCosineScheduler:
    """Test CosineScheduler functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the CosineScheduler for testing."""
        self.decay_steps = 100
        self.min_decay = 0.1
        self.scheduler = CosineScheduler(decay_steps=self.decay_steps,
                                         min_decay=self.min_decay)

    def test_cosine_scheduler_start(self) -> None:
        """Test CosineScheduler at the start of the schedule."""
        base_lr = 1.0
        lr_epoch_0 = self.scheduler(base_lr, 0)
        assert lr_epoch_0 == base_lr

    def test_cosine_scheduler_mid(self) -> None:
        """Test CosineScheduler midway through the schedule."""
        base_lr = 1.0
        epoch_mid = self.decay_steps // 2
        lr_epoch_mid = self.scheduler(base_lr, epoch_mid)
        base_term = self.min_decay * base_lr
        delta_term = base_lr - self.min_decay * base_lr
        cosine_decay = 1 / 2
        expected_lr_mid = base_term + delta_term * cosine_decay
        assert np.isclose(lr_epoch_mid, expected_lr_mid, rtol=1e-5)

    def test_cosine_scheduler_end(self) -> None:
        """Test CosineScheduler at the end of the schedule."""
        base_lr = 1.0
        lr_epoch_end = self.scheduler(base_lr, self.decay_steps)
        assert lr_epoch_end == self.min_decay * base_lr

    def test_cosine_scheduler_beyond_end(self) -> None:
        """Test CosineScheduler beyond decay_steps remains constant."""
        base_lr = 1.0
        lr_beyond_decay = self.scheduler(base_lr, self.decay_steps + 10)
        assert lr_beyond_decay == self.min_decay * base_lr

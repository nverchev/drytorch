"""Tests for the "schedulers" module."""

import pytest

from drytorch.lib.schedulers import (
    AbstractScheduler,
    ConstantScheduler,
    CosineScheduler,
    ExponentialScheduler,
    PolynomialScheduler,
    StepScheduler,
    rescale,
    restart,
    warmup,
)


class TestConstantScheduler:
    """Test the ConstantScheduler class."""

    @pytest.fixture
    def scheduler(self) -> AbstractScheduler:
        """Set up the instance."""
        return ConstantScheduler()

    def test_constant_scheduler(self, scheduler) -> None:
        """Test that ConstantScheduler returns the same learning rate."""
        base_lr = 0.1
        epochs = [0, 10, 50]

        for epoch in epochs:
            assert scheduler(base_lr, epoch) == base_lr


class TestExponentialScheduler:
    """Test the ExponentialScheduler class."""

    @pytest.fixture
    def exp_decay(self) -> float:
        """Return test argument."""
        return 0.9

    @pytest.fixture
    def min_decay(self) -> float:
        """Return test argument."""
        return 0.1

    @pytest.fixture
    def scheduler(self, exp_decay, min_decay) -> AbstractScheduler:
        """Set up the instance."""
        return ExponentialScheduler(exp_decay=exp_decay, min_decay=min_decay)

    def test_scheduler(self, scheduler, exp_decay, min_decay) -> None:
        """Test that the scheduler correctly decays learning rate."""
        base_lr = 1.0

        assert scheduler(base_lr, 0) == base_lr
        assert scheduler(base_lr, 1) == 0.91
        assert scheduler(base_lr, 22) == pytest.approx(0.1886293)
        assert scheduler(base_lr, 50) == pytest.approx(0.1046383)

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            ExponentialScheduler(exp_decay=0.0)
        with pytest.raises(ValueError):
            ExponentialScheduler(exp_decay=1.1)
        with pytest.raises(ValueError):
            ExponentialScheduler(min_decay=-0.1)
        with pytest.raises(ValueError):
            ExponentialScheduler(min_decay=1.1)


class TestCosineScheduler:
    """Test the CosineScheduler class."""

    @pytest.fixture
    def decay_steps(self) -> int:
        """Return test argument."""
        return 100

    @pytest.fixture
    def min_decay(self) -> float:
        """Return test argument."""
        return 0.1

    @pytest.fixture
    def scheduler(self, decay_steps, min_decay) -> AbstractScheduler:
        """Set up the instance."""
        return CosineScheduler(decay_steps=decay_steps, min_decay=min_decay)

    def test_cosine_scheduler_start(self, scheduler) -> None:
        """Test CosineScheduler at the start of the schedule."""
        base_lr = 1.0
        lr_epoch_0 = scheduler(base_lr, 0)
        assert lr_epoch_0 == base_lr

    def test_cosine_scheduler_mid(
        self, scheduler, decay_steps, min_decay
    ) -> None:
        """Test CosineScheduler midway through the schedule."""
        base_lr = 1.0
        epoch_mid = decay_steps // 2
        assert scheduler(base_lr, epoch_mid) == 0.55

    def test_cosine_scheduler_end(
        self, scheduler, decay_steps, min_decay
    ) -> None:
        """Test CosineScheduler at the end of the schedule."""
        base_lr = 1.0
        lr_epoch_end = scheduler(base_lr, decay_steps)
        assert lr_epoch_end == min_decay * base_lr

    def test_cosine_scheduler_beyond_end(
        self, scheduler, decay_steps, min_decay
    ) -> None:
        """Test CosineScheduler beyond decay_steps remains constant."""
        base_lr = 1.0
        lr_beyond_decay = scheduler(base_lr, decay_steps + 10)
        assert lr_beyond_decay == pytest.approx(min_decay * base_lr)

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            CosineScheduler(decay_steps=0)
        with pytest.raises(ValueError):
            CosineScheduler(min_decay=-0.1)
        with pytest.raises(ValueError):
            CosineScheduler(min_decay=1.1)


class TestPolynomialScheduler:
    """Test the PolynomialScheduler class."""

    @pytest.fixture
    def max_epochs(self) -> int:
        """Return test argument."""
        return 100

    @pytest.fixture
    def power(self) -> float:
        """Return test argument."""
        return 0.5

    @pytest.fixture
    def min_decay(self) -> float:
        """Return test argument."""
        return 0.1

    @pytest.fixture
    def scheduler(self, max_epochs, power, min_decay) -> AbstractScheduler:
        """Set up the instance."""
        return PolynomialScheduler(
            max_epochs=max_epochs, power=power, min_decay=min_decay
        )

    def test_polynomial_scheduler_start(self, scheduler) -> None:
        """Test PolynomialScheduler at the start of the schedule."""
        base_lr = 1.0
        assert scheduler(base_lr, 0) == base_lr

    def test_polynomial_scheduler_mid(
        self, scheduler, max_epochs, power, min_decay
    ) -> None:
        """Test PolynomialScheduler midway through the schedule."""
        base_lr = 1.0
        epoch = max_epochs // 2
        assert scheduler(base_lr, epoch) == pytest.approx(0.7363961)

    def test_polynomial_scheduler_end(
        self, scheduler, max_epochs, min_decay
    ) -> None:
        """Test PolynomialScheduler at the end of the schedule."""
        base_lr = 1.0
        assert scheduler(base_lr, max_epochs) == 0.1

    def test_polynomial_scheduler_beyond_end(
        self, scheduler, max_epochs, min_decay
    ) -> None:
        """Test PolynomialScheduler beyond max_epochs remains at min_decay."""
        base_lr = 1.0
        assert scheduler(base_lr, max_epochs + 10) == min_decay * base_lr

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            PolynomialScheduler(max_epochs=0)
        with pytest.raises(ValueError):
            PolynomialScheduler(power=-0.1)
        with pytest.raises(ValueError):
            PolynomialScheduler(min_decay=-0.1)
        with pytest.raises(ValueError):
            PolynomialScheduler(min_decay=1.1)


class TestStepScheduler:
    """Test the StepScheduler class."""

    @pytest.fixture
    def milestones(self) -> list[int]:
        """Return test argument."""
        return [50, 100, 150]

    @pytest.fixture
    def gamma(self) -> float:
        """Return test argument."""
        return 0.1

    @pytest.fixture
    def scheduler(self, milestones, gamma) -> AbstractScheduler:
        """Set up the instance."""
        return StepScheduler(milestones=milestones, gamma=gamma)

    def test_step_scheduler_before_first_milestone(self, scheduler) -> None:
        """Test StepScheduler before any milestones."""
        base_lr = 1.0
        assert scheduler(base_lr, 49) == base_lr

    def test_step_scheduler_at_first_milestone(self, scheduler, gamma) -> None:
        """Test StepScheduler at the first milestone."""
        base_lr = 1.0
        assert scheduler(base_lr, 50) == base_lr * gamma

    def test_step_scheduler_between_milestones(self, scheduler, gamma) -> None:
        """Test StepScheduler between milestones."""
        base_lr = 1.0
        assert scheduler(base_lr, 75) == base_lr * gamma
        assert scheduler(base_lr, 120) == base_lr * (gamma**2)

    def test_step_scheduler_at_multiple_milestones(
        self, scheduler, gamma
    ) -> None:
        """Test StepScheduler at multiple milestones."""
        base_lr = 1.0
        assert scheduler(base_lr, 100) == base_lr * (gamma**2)
        assert scheduler(base_lr, 150) == base_lr * (gamma**3)

    def test_step_scheduler_beyond_last_milestone(
        self, scheduler, gamma
    ) -> None:
        """Test StepScheduler beyond the last milestone."""
        base_lr = 1.0
        assert scheduler(base_lr, 200) == base_lr * (gamma**3)

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            StepScheduler(milestones=[-10, 20])
        with pytest.raises(ValueError):
            StepScheduler(milestones=[0, 20])
        with pytest.raises(ValueError):
            StepScheduler(milestones=[100, 50])
        with pytest.raises(ValueError):
            StepScheduler(gamma=0.0)
        with pytest.raises(ValueError):
            StepScheduler(gamma=1.1)


class TestBindingOperation:
    """Test the generic binding operation on AbstractScheduler."""

    @pytest.fixture
    def initial_scheduler(self) -> AbstractScheduler:
        """Set up the instance."""
        return ConstantScheduler()

    @pytest.fixture
    def base_lr(self) -> float:
        """Return test argument."""
        return 1.0

    def test_binding_with_rescale(self, initial_scheduler, base_lr) -> None:
        """Test binding a scaling transformation."""
        factor = 2.0
        scaled_scheduler = initial_scheduler.bind(rescale(factor))
        epoch = 5
        assert scaled_scheduler(base_lr, epoch) == factor * base_lr

    def test_binding_with_warmup(self, initial_scheduler, base_lr) -> None:
        """Test binding a warmup transformation."""
        warmup_steps = 3
        warmed_up_scheduler = initial_scheduler.bind(warmup(warmup_steps))
        expected = base_lr * (1 / warmup_steps)
        assert warmed_up_scheduler(base_lr, 1) == expected
        assert warmed_up_scheduler(base_lr, warmup_steps) == base_lr

    def test_binding_with_restart(self, base_lr) -> None:
        """Test binding a restart transformation."""
        base_scheduler = CosineScheduler(decay_steps=10, min_decay=0.1)
        restart_interval = 10
        restart_fraction = 0.2

        restarted_scheduler = base_scheduler.bind(
            restart(restart_interval, restart_fraction)
        )
        expected = base_scheduler(base_lr, 5)
        assert restarted_scheduler(base_lr, 5) == expected

        expected = base_scheduler(base_lr, 10)
        assert restarted_scheduler(base_lr, 10) == expected

        expected = base_scheduler(base_lr * restart_fraction, 5)
        assert restarted_scheduler(base_lr, 15) == expected

    def test_binding_chaining(self, initial_scheduler, base_lr) -> None:
        """Test chaining multiple binding operations."""
        chained_scheduler = initial_scheduler.bind(warmup(warmup_steps=2))
        chained_scheduler = chained_scheduler.bind(rescale(factor=0.5))

        assert chained_scheduler(base_lr, 0) == 0.0
        assert chained_scheduler(base_lr, 1) == 0.25
        assert chained_scheduler(base_lr, 2) == 0.5


class TestRescaleLogic:
    """Test the rescales transformation via bind."""

    def test_scaled_scheduler(self) -> None:
        """Test it correctly scales the base scheduler's output."""
        factor = 0.5
        base_lr = 0.1
        scheduler = ConstantScheduler().bind(rescale(factor))

        assert scheduler(base_lr, 10) == factor * base_lr
        assert scheduler.logic.factor == factor

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            ConstantScheduler().bind(rescale(0.0))


class TestRestartLogic:
    """Test the restart transformation via bind."""

    def test_restart_scheduler(self) -> None:
        """Test periodic restarts of the base scheduler."""
        interval, fraction, max_r = 50, 0.5, 3
        base_lr = 1.0

        scheduler = CosineScheduler(decay_steps=50, min_decay=0.01).bind(
            restart(
                restart_interval=interval,
                restart_fraction=fraction,
                max_restart=max_r,
            )
        )

        expected = CosineScheduler(decay_steps=50, min_decay=0.01)(
            base_lr * fraction, 1
        )
        assert scheduler(base_lr, 51) == expected

        assert scheduler.logic.restart_interval == interval
        assert scheduler.logic.restart_fraction == fraction
        assert scheduler.logic.max_restart == max_r

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            ConstantScheduler().bind(restart(restart_interval=0))


class TestWarmupLogic:
    """Test the warmup transformation via bind."""

    def test_warmup_scheduler(self) -> None:
        """Test linear warmup phase before the base scheduler starts."""
        steps = 10
        base_lr = 0.1

        scheduler = ConstantScheduler().bind(warmup(warmup_steps=steps))

        assert scheduler(base_lr, 0) == 0.0
        assert scheduler(base_lr, 5) == pytest.approx(0.05)
        assert scheduler(base_lr, 10) == base_lr
        assert scheduler(base_lr, 15) == base_lr
        assert scheduler.logic.warmup_steps == steps

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            ConstantScheduler().bind(warmup(warmup_steps=-1))

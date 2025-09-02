"""Tests for the "schedulers" module."""

import pytest

from drytorch.lib.schedulers import (
    AbstractScheduler,
    ConstantScheduler,
    CosineScheduler,
    ExponentialScheduler,
    PolynomialScheduler,
    RescaleScheduler,
    RestartScheduler,
    StepScheduler,
    WarmupScheduler,
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


class TestScaledScheduler:
    """Test the RescaledScheduler class."""

    @pytest.fixture
    def factor(self) -> float:
        """Return test argument."""
        return 0.5

    @pytest.fixture
    def base_scheduler(self) -> AbstractScheduler:
        """Return test argument."""
        return ConstantScheduler()

    @pytest.fixture
    def scheduler(self, factor, base_scheduler) -> AbstractScheduler:
        """Set up the instance."""
        return RescaleScheduler(factor=factor, base_scheduler=base_scheduler)

    def test_scaled_scheduler(self, scheduler, factor, base_scheduler) -> None:
        """Test it correctly scales the base scheduler's output."""
        base_lr = 0.1
        epoch = 10
        expected_lr = factor * base_scheduler(base_lr, epoch)
        assert scheduler(base_lr, epoch) == expected_lr

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            RescaleScheduler(factor=0.0, base_scheduler=ConstantScheduler())
        with pytest.raises(ValueError):
            RescaleScheduler(factor=-1.0, base_scheduler=ConstantScheduler())


class TestRestartScheduler:
    """Test the RestartScheduler class."""

    @pytest.fixture
    def restart_interval(self) -> int:
        """Return test argument."""
        return 50

    @pytest.fixture
    def restart_fraction(self) -> float:
        """Return test argument."""
        return 0.5

    @pytest.fixture
    def max_restart(self) -> int:
        """Return test argument."""
        return 3

    @pytest.fixture
    def base_scheduler(self) -> AbstractScheduler:
        """Return test argument."""
        return CosineScheduler(decay_steps=50, min_decay=0.01)

    @pytest.fixture
    def scheduler(
        self, base_scheduler, restart_interval, restart_fraction, max_restart
    ) -> AbstractScheduler:
        """Set up the instance."""
        return RestartScheduler(
            base_scheduler=base_scheduler,
            restart_interval=restart_interval,
            restart_fraction=restart_fraction,
            max_restart=max_restart,
        )

    def test_restart_scheduler_no_restart(
        self, scheduler, base_scheduler, restart_interval
    ) -> None:
        """Test RestartScheduler before the first restart."""
        base_lr = 1.0
        epoch = restart_interval - 10
        assert scheduler(base_lr, epoch) == base_scheduler(base_lr, epoch)

    def test_restart_scheduler_first_restart(
        self, scheduler, base_scheduler, restart_interval, restart_fraction
    ) -> None:
        """Test RestartScheduler exactly at the first restart point."""
        base_lr = 1.0
        epoch = restart_interval + 1
        expected = base_scheduler(base_lr * restart_fraction, 1)
        assert scheduler(base_lr, epoch) == expected

    def test_restart_scheduler_after_first_restart(
        self, scheduler, base_scheduler, restart_interval, max_restart
    ) -> None:
        """Test RestartScheduler after the first restart point."""
        base_lr = 1.0
        epoch = (max_restart + 1) * restart_interval
        expected = base_scheduler(base_lr, epoch)
        assert scheduler(base_lr, epoch) == expected

    def test_restart_scheduler_multiple_restarts(
        self, scheduler, base_scheduler, restart_interval, restart_fraction
    ) -> None:
        """Test RestartScheduler after multiple restarts."""
        base_lr = 1.0
        epoch = (2 * restart_interval) + 20
        expected_start_value = base_lr * restart_fraction
        expected = base_scheduler(expected_start_value, 20)
        assert scheduler(base_lr, epoch) == expected

    def test_restart_after_max_restarts(
        self, scheduler, base_scheduler, restart_interval, restart_fraction
    ) -> None:
        """Test RestartScheduler after multiple restarts."""
        base_lr = 1.0
        epoch = (2 * restart_interval) + 20
        expected_start_value = base_lr * restart_fraction
        expected = base_scheduler(expected_start_value, 20)
        assert scheduler(base_lr, epoch) == expected

    def test_invalid_params(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            RestartScheduler(ConstantScheduler(), restart_interval=0)
        with pytest.raises(ValueError):
            RestartScheduler(
                ConstantScheduler(), restart_interval=10, restart_fraction=0.0
            )
        with pytest.raises(ValueError):
            RestartScheduler(
                ConstantScheduler(), restart_interval=10, restart_fraction=-0.1
            )
            with pytest.raises(ValueError):
                RestartScheduler(
                    ConstantScheduler(), restart_interval=10, max_restart=-1
                )


class TestWarmupScheduler:
    """Test the WarmupScheduler class."""

    @pytest.fixture
    def warmup_steps(self) -> int:
        """Return test argument."""
        return 10

    @pytest.fixture
    def base_scheduler(self) -> AbstractScheduler:
        """Return test argument."""
        return ConstantScheduler()

    @pytest.fixture
    def scheduler(self, warmup_steps, base_scheduler) -> AbstractScheduler:
        """Set up the instance."""
        return WarmupScheduler(
            warmup_steps=warmup_steps, base_scheduler=base_scheduler
        )

    def test_warmup_start(self, scheduler) -> None:
        """Test that warmup starts at zero learning rate."""
        base_lr = 0.1
        assert scheduler(base_lr, 0) == 0.0

    def test_warmup_middle(self, scheduler, warmup_steps) -> None:
        """Test that warmup increases linearly."""
        base_lr = 0.1
        mid_warmup = warmup_steps // 2
        expected_lr = base_lr * (mid_warmup / warmup_steps)
        assert scheduler(base_lr, mid_warmup) == expected_lr

    def test_warmup_end(self, scheduler, warmup_steps) -> None:
        """Test that warmup reaches base_lr."""
        base_lr = 0.1
        end_warmup_lr = scheduler(base_lr, warmup_steps)
        assert end_warmup_lr == base_lr

    def test_post_warmup(self, scheduler, base_scheduler, warmup_steps) -> None:
        """Test that the base scheduler behaves correctly after warmup."""
        base_lr = 0.1
        epochs_after_warmup = 5
        total_epochs = warmup_steps + epochs_after_warmup
        post_warmup_lr = scheduler(base_lr, total_epochs)
        expected_lr = base_scheduler(base_lr, epochs_after_warmup)
        assert post_warmup_lr == expected_lr

    def test_invalid_params(self, base_scheduler) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            WarmupScheduler(base_scheduler, warmup_steps=-1)


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

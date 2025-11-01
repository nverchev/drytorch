"""Tests for the "gradient_ops" module."""

from collections.abc import Iterable

import torch

import pytest

from drytorch.lib.gradient_ops import (
    EMACriterion,
    GradNormClipper,
    GradParamNormalizer,
    GradValueClipper,
    GradZScoreNormalizer,
    HistClipper,
    ParamHistClipper,
    StatsCollector,
    ZStatCriterion,
    max_clipping,
    mean_clipping,
    reciprocal_clipping,
)


@pytest.fixture
def tensor_no_grad() -> torch.nn.Parameter:
    """Tensor with no gradient."""
    return torch.nn.Parameter(torch.ones(3, 3))


@pytest.fixture
def tensor_zero_grad(tensor_no_grad) -> torch.nn.Parameter:
    """Tensor with zero gradient."""
    tensor_zero_grad = torch.nn.Parameter(torch.ones(3, 3))
    tensor_zero_grad.grad = torch.zeros(3, 3)
    return tensor_zero_grad


@pytest.fixture
def tensor_random_grad(tensor_no_grad) -> torch.nn.Parameter:
    """Tensor with a random gradient."""
    tensor_zero_grad = torch.nn.Parameter(torch.ones(3, 3))
    tensor_zero_grad.grad = torch.randn(3, 3)
    return tensor_zero_grad


@pytest.fixture
def example_parameters(
    tensor_no_grad, tensor_zero_grad, tensor_random_grad
) -> Iterable[torch.nn.Parameter]:
    """Iterator of tensors with: no grad, zero grad, random grad."""
    # create an iterator that will be exhausted as in model.parameters()
    tensor_tuple = (tensor_no_grad, tensor_zero_grad, tensor_random_grad)
    return (tensor for tensor in tensor_tuple)


class TestGradNormalizer:
    """Test the GradNormalizer class."""

    @pytest.fixture
    def grad_op(self):
        """Set up the instance."""
        return GradParamNormalizer()

    def test_call(
        self, grad_op, tensor_zero_grad, tensor_random_grad, example_parameters
    ) -> None:
        """Test call functionality."""
        grad_op(example_parameters)
        assert not tensor_zero_grad.grad.any()
        assert tensor_random_grad.grad.norm(2).item() == pytest.approx(1.0)


class TestGradZScoreNormalizer:
    """Test the TestGradZScoreNormalizer class."""

    @pytest.fixture
    def grad_op(self):
        """Set up the instance."""
        return GradZScoreNormalizer()

    def test_call(
        self, grad_op, tensor_zero_grad, tensor_random_grad, example_parameters
    ) -> None:
        """Test call functionality."""
        grad_op(example_parameters)
        assert not tensor_zero_grad.grad.any()

        mean = tensor_random_grad.grad.mean().item()
        var = tensor_random_grad.grad.var().item()
        assert mean == pytest.approx(0.0, abs=1e-6)
        assert var == pytest.approx(1.0, abs=1e-6)


class TestGradNormClipper:
    """Test the GradNormClipper class."""

    @pytest.fixture
    def threshold(self):
        """Instance argument."""
        return 0.01

    @pytest.fixture
    def grad_op(self, threshold):
        """Set up the instance."""
        return GradNormClipper(threshold)

    def test_wrong_threshold(self) -> None:
        """Test threshold of 0 returns an error."""
        with pytest.raises(ValueError):
            GradNormClipper(0)

    def test_call(
        self, threshold, grad_op, tensor_random_grad, example_parameters
    ) -> None:
        """Test call functionality."""
        grad_op(example_parameters)
        norm = tensor_random_grad.grad.norm().item()
        assert norm == pytest.approx(threshold, abs=1e-6)


class TestGradValueClipper:
    """Test the GradValueClipper class."""

    @pytest.fixture
    def threshold(self):
        """Instance argument."""
        return 0.01

    @pytest.fixture
    def grad_op(self, threshold):
        """Set up the instance."""
        return GradValueClipper(threshold)

    def test_wrong_threshold(self) -> None:
        """Test negative threshold returns error."""
        with pytest.raises(ValueError):
            GradValueClipper(-1)

    def test_call(
        self, threshold, grad_op, tensor_random_grad, example_parameters
    ) -> None:
        """Test call functionality."""
        grad_op(example_parameters)
        max_value = torch.max(torch.abs(tensor_random_grad.grad)).item()
        assert max_value == pytest.approx(threshold, abs=1e-6)


def test_reciprocal_clipping() -> None:
    """Test reciprocal_clipping behavior."""
    # continuous on the threshold
    expected = zt = z_thresh = 2.0
    assert reciprocal_clipping(zt, z_thresh) == expected

    # decrease after that
    assert reciprocal_clipping(zt + 1, z_thresh) < expected


def test_mean_clipping() -> None:
    """Test mean_clipping always returns 0.0."""
    zt = z_thresh = 2.0
    assert not mean_clipping(zt, z_thresh)


def test_max_clipping() -> None:
    """Test max_clipping always returns z_thresh."""
    zt = 5.0
    z_thresh = 2.0
    assert max_clipping(zt, z_thresh) == z_thresh


class TestEMACriterion:
    """Tests for the EMACriterion class."""

    @pytest.fixture
    def ema_criterion(self) -> EMACriterion:
        """Fixture for an EMACriterion instance."""
        return EMACriterion(alpha=0.5, r_thresh=2.0)

    def test_init(self, ema_criterion: EMACriterion) -> None:
        """Test initialization of EMACriterion."""
        assert ema_criterion.alpha == 0.5
        assert ema_criterion.r_thresh == 2.0
        assert ema_criterion.clipping_function == max_clipping

    def test_should_clip(self, ema_criterion) -> None:
        """Test should_clip logic."""
        # no clipping if mu_t is 0
        assert not ema_criterion.should_clip(10.0)

        # set a mean
        ema_criterion._mu_t = 5.0

        # value / mu_t = 1.8 < r_thresh (2.0)
        assert not ema_criterion.should_clip(9.0)

        # value / mu_t = 2.02 > r_thresh (2.0)
        assert ema_criterion.should_clip(10.1)

        # value / mu_t = 2.0 == r_thresh (2.0) - check boundary
        assert not ema_criterion.should_clip(10.0)

    def test_get_clip_value(self, ema_criterion: EMACriterion) -> None:
        """Test get_clip_value logic."""
        # returns value if mu_t is 0
        assert ema_criterion.get_clip_value(10.0) == 10.0

        # set a mean
        ema_criterion._mu_t = 5.0

        # max_clipping
        assert ema_criterion.get_clip_value(10.1) == 10.0

    def test_update(self, ema_criterion: EMACriterion) -> None:
        """Test update of mu_t."""
        ema_criterion.update(10.0)  # mu_t = 0.5 * 0 + 0.5 * 10 = 5.0
        assert ema_criterion._mu_t == 5.0

    def test_set_statistics(self, ema_criterion: EMACriterion) -> None:
        """Test setting initial statistics."""
        ema_criterion.set_statistics(mean=100.0)
        assert ema_criterion._mu_t == 100.0

    def test_reset(self, ema_criterion: EMACriterion) -> None:
        """Test resetting the criterion."""
        ema_criterion._mu_t = 50.0
        ema_criterion.reset()
        assert ema_criterion._mu_t == 0.0


class TestZStatCriterion:
    """Tests for the ZStatCriterion class."""

    @pytest.fixture
    def zstat_criterion(self) -> ZStatCriterion:
        """Fixture for a ZStatCriterion instance."""
        return ZStatCriterion(alpha=0.9, z_thresh=2.0, eps=1e-6)

    def test_init(self, zstat_criterion: ZStatCriterion) -> None:
        """Test initialization of ZStatCriterion."""
        assert zstat_criterion.alpha == 0.9
        assert zstat_criterion.z_thresh == 2.0
        assert zstat_criterion.clipping_function == reciprocal_clipping

    def test_should_clip(self, zstat_criterion: ZStatCriterion) -> None:
        """Test should_clip logic."""
        # no clipping if mu_t is 0
        assert not zstat_criterion.should_clip(10.0)

        # set initial stats
        zstat_criterion._mu_t = 10.0
        zstat_criterion._v_t = 4.0  # std_dev = 2.0

        # Z-score = (value - 10) / 2
        assert not zstat_criterion.should_clip(12.0)  # Z-score = 1.0 < 2.0
        assert zstat_criterion.should_clip(16.0)  # Z-score = 3.0 > 2.0
        assert zstat_criterion.should_clip(4)  # abs(Z-score) = 3 > 2.0

    def test_get_clip_value(self, zstat_criterion: ZStatCriterion) -> None:
        """Test get_clip_value logic."""
        # returns value if mu_t is 0
        assert zstat_criterion.get_clip_value(10.0) == 10.0

        # set initial stats
        zstat_criterion._mu_t = 10.0
        zstat_criterion._v_t = 1.0

        # If not clipping, return the original value
        assert zstat_criterion.get_clip_value(11.0) == 11.0  # Z-score = 1.0

        # reciprocal_clipping
        # value = 15.0, Z-score = (15-10)/1 = 5.0
        # new_z_score = reciprocal_clipping(5.0, 2.0) = 2.0**2 / 5.0 = 0.8
        # clip_value = 10.0 + 0.8 * 1.0 = 10.8
        assert zstat_criterion.get_clip_value(15.0) == pytest.approx(10.8)

    def test_update(self, zstat_criterion: ZStatCriterion) -> None:
        """Test update of mu_t and v_t."""
        # initial: mu_t = 0.0, v_t = 1.0
        zstat_criterion.update(10.0)
        # mu_t = 0.9 * 0 + 0.1 * 10 = 1.0
        # variance = (10 - 0.0)**2 = 100  # Use OLD mean (0.0)
        # v_t = 0.9 * 1 + 0.1 * 100 = 0.9 + 10.0 = 10.9
        assert zstat_criterion._mu_t == pytest.approx(1.0)
        assert zstat_criterion._v_t == pytest.approx(10.9)

        zstat_criterion.update(20.0)
        # mu_t = 0.9 * 1.0 + 0.1 * 20 = 0.9 + 2.0 = 2.9
        # variance = (20 - 1.0)**2 = 19.0**2 = 361  # Use OLD mean (1.0)
        # v_t = 0.9 * 10.9 + 0.1 * 361 = 9.81 + 36.1 = 45.91
        assert zstat_criterion._mu_t == pytest.approx(2.9)
        assert zstat_criterion._v_t == pytest.approx(45.91)

    def test_set_statistics(self, zstat_criterion: ZStatCriterion) -> None:
        """Test setting initial statistics."""
        zstat_criterion.set_statistics(mean=50.0, variance=10.0)
        assert zstat_criterion._mu_t == 50.0
        assert zstat_criterion._v_t == 10.0

        # variance is not updated by default
        zstat_criterion.set_statistics(mean=50.0)
        assert zstat_criterion._mu_t == 50.0
        assert zstat_criterion._v_t == 10.0

    def test_reset(self, zstat_criterion: ZStatCriterion) -> None:
        """Test resetting the criterion."""
        zstat_criterion._mu_t = 50.0
        zstat_criterion._v_t = 10.0
        zstat_criterion.reset()
        assert zstat_criterion._mu_t == 0.0
        assert zstat_criterion._v_t == 1.0


class TestStatsCollector:
    """Tests for the StatsCollector class."""

    @pytest.fixture
    def stats_collector(self) -> StatsCollector:
        """Fixture for a StatsCollector instance."""
        return StatsCollector(max_samples=3)

    def test_init(self, stats_collector: StatsCollector) -> None:
        """Test initialization of StatsCollector."""
        assert stats_collector.max_samples == 3
        assert stats_collector.active is True
        assert len(stats_collector) == 0

    def test_mean(self, stats_collector: StatsCollector) -> None:
        """Test mean property."""
        assert stats_collector.mean == 0.0  # Empty data
        stats_collector.append(10.0)
        assert stats_collector.mean == 10.0
        stats_collector.append(20.0)
        stats_collector.append(30.0)
        assert stats_collector.mean == 20.0

    def test_variance(self, stats_collector: StatsCollector) -> None:
        """Test variance property."""
        assert stats_collector.variance == 1.0  # less than 2 samples
        stats_collector.append(10.0)
        assert stats_collector.variance == 1.0  # less than 2 samples
        stats_collector.append(20.0)
        # (10-15)^2 + (20-15)^2 / (2-1) = (-5)^2 + (5)^2 = 25 + 25 = 50
        assert stats_collector.variance == 50.0
        stats_collector.append(30.0)
        # (10-20)^2 + (20-20)^2 + (30-20)^2 / (3-1) = 100 + 0 + 100 / 2 = 100
        assert stats_collector.variance == 100.0

    def test_is_complete(self, stats_collector: StatsCollector) -> None:
        """Test is_complete method."""
        assert not stats_collector.is_complete()
        stats_collector.append(1.0)
        stats_collector.append(2.0)
        assert not stats_collector.is_complete()
        stats_collector.append(3.0)
        assert stats_collector.is_complete()
        assert not stats_collector.active
        stats_collector.append(4.0)  # should not add more than max_samples
        assert stats_collector.is_complete()
        assert len(stats_collector) == 3

    def test_append(self, stats_collector: StatsCollector) -> None:
        """Test append method."""
        stats_collector.append(1.0)
        assert stats_collector._data == [1.0]
        stats_collector.append(2.0)
        stats_collector.append(3.0)
        assert stats_collector._data == [1.0, 2.0, 3.0]
        stats_collector.append(4.0)  # should not append beyond max_samples
        assert stats_collector._data == [1.0, 2.0, 3.0]

    def test_reset(self, stats_collector: StatsCollector) -> None:
        """Test reset method."""
        stats_collector.append(1.0)
        stats_collector.active = False
        stats_collector.reset()
        assert not len(stats_collector)
        assert stats_collector.active is True


class TestHistClipping:
    """Tests for the HistClipping class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests for the class."""
        self.mock_get_clip_value = mocker.patch.object(
            EMACriterion, 'get_clip_value'
        )
        self.mock_warmup_reset = mocker.patch.object(EMACriterion, 'reset')
        self.mock_set_statistics = mocker.patch.object(
            EMACriterion, 'set_statistics'
        )
        self.mock_should_clip = mocker.patch.object(EMACriterion, 'should_clip')
        self.mock_update = mocker.patch.object(EMACriterion, 'update')
        self.mock_init_clip_call = mocker.patch.object(
            GradNormClipper, '__call__'
        )
        self.mock_criterion_reset = mocker.patch.object(StatsCollector, 'reset')
        self.mock_torch_clip = mocker.patch.object(
            torch.nn.utils, 'clip_grad_norm_'
        )
        return

    @pytest.fixture
    def grad_clipping(self) -> HistClipper:
        """Set up a test instance."""
        # criterion clips when higher than mean
        criterion = EMACriterion(alpha=0.9, r_thresh=1.0)
        return HistClipper(criterion=criterion, n_warmup_steps=1)

    def test_init(self, grad_clipping: HistClipper) -> None:
        """Test initialization."""
        assert isinstance(grad_clipping.criterion, EMACriterion)
        assert isinstance(grad_clipping.warmup_clip_strategy, GradNormClipper)
        assert grad_clipping.n_warmup_steps == 1

    def test_call_warmup_phase(self, grad_clipping, example_parameters) -> None:
        """Test behavior during the warmup phase."""
        list_params = list(example_parameters)
        grad_clipping(list_params)
        assert len(grad_clipping._warmup_handler) == 1
        self.mock_init_clip_call.assert_called_once()
        self.mock_set_statistics.assert_not_called()

        self.mock_init_clip_call.reset_mock()
        grad_clipping(list_params)
        self.mock_init_clip_call.assert_not_called()
        self.mock_set_statistics.assert_called_once()
        assert len(grad_clipping._warmup_handler) == 1
        assert not grad_clipping._warmup_handler.active
        self.mock_should_clip.assert_called_once()

    def test_call_after_warmup(self, grad_clipping, example_parameters) -> None:
        """Test behavior after the warmup phase, with and without clipping."""
        # skip warm up
        grad_clipping._warmup_handler.active = False
        self.mock_should_clip.return_value = False
        grad_clipping(example_parameters)
        self.mock_get_clip_value.assert_not_called()
        self.mock_torch_clip.assert_not_called()
        self.mock_update.assert_called_once()

        self.mock_update.reset_mock()
        self.mock_should_clip.return_value = True
        grad_clipping(example_parameters)
        self.mock_get_clip_value.assert_called_once()
        self.mock_torch_clip.assert_called_once()
        self.mock_update.assert_called_once()

    def test_reset(self, grad_clipping: HistClipper) -> None:
        """Test reset method."""
        grad_clipping.reset()
        self.mock_warmup_reset.assert_called_once()
        self.mock_criterion_reset.assert_called_once()


class TestParamHistClipping:
    """Tests for the ParamHistClipping class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests for the class."""
        self.mock_get_clip_value = mocker.patch.object(
            ZStatCriterion, 'get_clip_value'
        )
        self.mock_warmup_reset = mocker.patch.object(ZStatCriterion, 'reset')
        self.mock_set_statistics = mocker.patch.object(
            ZStatCriterion, 'set_statistics'
        )
        self.mock_should_clip = mocker.patch.object(
            ZStatCriterion, 'should_clip'
        )
        self.mock_update = mocker.patch.object(ZStatCriterion, 'update')
        self.mock_init_clip_call = mocker.patch.object(
            GradNormClipper, '__call__'
        )
        self.mock_criterion_reset = mocker.patch.object(StatsCollector, 'reset')
        self.mock_torch_clip = mocker.patch.object(
            torch.nn.utils, 'clip_grad_norm_'
        )
        return

    @pytest.fixture
    def grad_clipping(self) -> ParamHistClipper:
        """Set up a test instance."""
        criterion = ZStatCriterion(alpha=0.9, z_thresh=1.0)
        return ParamHistClipper(criterion=criterion, n_warmup_steps=1)

    def test_init(self, grad_clipping: HistClipper) -> None:
        """Test initialization."""
        assert isinstance(grad_clipping.criterion, ZStatCriterion)
        assert isinstance(grad_clipping.warmup_clip_strategy, GradNormClipper)
        assert grad_clipping.n_warmup_steps == 1

    def test_call_warmup_phase(self, grad_clipping, example_parameters) -> None:
        """Test behavior during the warmup phase."""
        list_params = [p for p in example_parameters if p.grad is not None]
        grad_clipping(list_params)
        assert len(grad_clipping._dict_warmup_handler) == len(list_params)
        assert len(grad_clipping._dict_criterion) == len(list_params)
        assert self.mock_init_clip_call.call_count == len(list_params)
        self.mock_set_statistics.assert_not_called()

        self.mock_init_clip_call.reset_mock()
        grad_clipping(list_params)

        # after the second call, warmup should be complete
        self.mock_init_clip_call.assert_not_called()
        assert self.mock_set_statistics.call_count == len(list_params)
        for handler in grad_clipping._dict_warmup_handler.values():
            assert not handler.active
        assert self.mock_should_clip.call_count == len(list_params)

    def test_call_after_warmup(self, grad_clipping, example_parameters) -> None:
        """Test behavior after the warmup phase, with and without clipping."""
        list_params = [p for p in example_parameters if p.grad is not None]

        # skip warmup by making handlers inactive
        for param in list_params:
            param_id = id(param)
            grad_clipping._dict_warmup_handler[param_id].active = False

        self.mock_should_clip.return_value = False
        grad_clipping(list_params)
        self.mock_get_clip_value.assert_not_called()
        self.mock_torch_clip.assert_not_called()
        assert self.mock_update.call_count == len(list_params)

        self.mock_update.reset_mock()
        self.mock_should_clip.return_value = True
        grad_clipping(list_params)
        assert self.mock_get_clip_value.call_count == len(list_params)
        assert self.mock_torch_clip.call_count == len(list_params)
        assert self.mock_update.call_count == len(list_params)

    def test_reset(self, grad_clipping: ParamHistClipper) -> None:
        """Test reset method."""
        # add some entries to the dictionaries first
        grad_clipping._dict_warmup_handler[1] = StatsCollector(1)
        grad_clipping._dict_criterion[1] = ZStatCriterion()

        grad_clipping.reset()

        # Check that dictionaries are cleared
        assert len(grad_clipping._dict_warmup_handler) == 0
        assert len(grad_clipping._dict_criterion) == 0

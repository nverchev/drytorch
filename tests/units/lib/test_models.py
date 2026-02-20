"""Tests for the "models" module."""

import torch

import pytest

from drytorch import Model
from drytorch.core import exceptions
from drytorch.lib.models import EMAModel, ModelOptimizer, SWAModel


@pytest.fixture(autouse=True, scope='module')
def setup_module(session_mocker) -> None:
    """Fixture for a mock experiment."""
    session_mocker.patch('drytorch.core.register.register_model')
    return


class ComplexModule(torch.nn.Module):
    """Example for an arbitrarily complex module."""

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(2, 1), torch.nn.Linear(1, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.linear2(self.relu(self.linear(inputs)))


class TestModel:
    """Tests for the Model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Set up torch.autocast mocks."""
        self.mock_autocast = mocker.patch('torch.autocast')
        self.mock_context = mocker.Mock()
        self.mock_autocast.return_value.__enter__ = mocker.Mock(
            return_value=self.mock_context
        )
        self.mock_autocast.return_value.__exit__ = mocker.Mock(
            return_value=None
        )

    @pytest.fixture(scope='class')
    def complex_model(self) -> Model[torch.Tensor, torch.Tensor]:
        """Fixture of a complex model wrapped with Model."""
        return Model(ComplexModule(), name='complex_model')

    @pytest.fixture(scope='class')
    def complex_dist_model(self) -> Model[torch.Tensor, torch.Tensor]:
        """Fixture of a complex model wrapped with Model."""
        return Model(torch.nn.DataParallel(ComplexModule()), name='dist_model')

    def test_model_increment_epoch(self, complex_model: Model) -> None:
        """Test Model's increment_epoch method increases the epoch count."""
        complex_model.increment_epoch()
        assert complex_model.epoch == 1

    def test_no_dist(self, complex_model: Model) -> None:
        """Test module is exec_module outside distributed settings."""
        assert complex_model.module is complex_model.exec_module

    def test_module_returns_unwrapped(self, complex_dist_model: Model) -> None:
        """Test that module property returns the raw nn.Module."""
        assert not isinstance(complex_dist_model.module, torch.nn.DataParallel)


class TestSWAModel:
    """Tests for the SWAModel wrapper."""

    @pytest.fixture
    def swa_model(self) -> SWAModel[torch.Tensor, torch.Tensor]:
        """Fixture for AveragedModel."""
        model = SWAModel(ComplexModule(), name='swa_model', start_epoch=2)
        model.epoch = 2
        return model

    def test_init(self, swa_model) -> None:
        """Averaged module should exist and be frozen."""
        assert swa_model.start_epoch == 2
        for param in swa_model.averaged_module.parameters():
            assert param.requires_grad is False

    def test_update_parameters(self, swa_model) -> None:
        """Calling update_parameters should modify averaged weights."""
        initial = [p.clone() for p in swa_model.averaged_module.parameters()]
        for p in swa_model.module.parameters():
            p.data.zero_()

        swa_model.post_epoch_update()

        updated = list(swa_model.averaged_module.parameters())

        assert any(
            not torch.allclose(p0, p1)
            for p0, p1 in zip(initial, updated, strict=False)
        )

    def test_update_parameters_no_start(self, swa_model) -> None:
        """No averaging should occur if start_epoch is not reached yet."""
        swa_model.epoch = 0
        initial = [p.clone() for p in swa_model.averaged_module.parameters()]
        for p in swa_model.module.parameters():
            p.data.zero_()

        swa_model.post_epoch_update()

        updated = list(swa_model.averaged_module.parameters())

        assert all(
            torch.allclose(p0, p1)
            for p0, p1 in zip(initial, updated, strict=False)
        )

    def test_call_uses_base_model_when_not_inference(
        self, swa_model, mocker
    ) -> None:
        """Forward should use the base model outside inference mode."""
        spy = mocker.spy(swa_model.module, 'forward')

        x = torch.randn(1, 1)
        swa_model(x)

        assert spy.called

    def test_call_uses_averaged_model_in_inference(
        self, swa_model, mocker
    ) -> None:
        """Forward should use averaged model in inference mode."""
        spy = mocker.spy(swa_model.averaged_module, 'forward')

        x = torch.randn(1, 1)

        with torch.inference_mode():
            swa_model(x)

        assert spy.called

    def test_call_uses_base_model_before_start_epoch(
        self, swa_model, mocker
    ) -> None:
        """Forward should use averaged model in inference mode."""
        swa_model.epoch = swa_model.start_epoch - 1
        spy = mocker.spy(swa_model.module, 'forward')

        x = torch.randn(1, 1)

        with torch.inference_mode():
            swa_model(x)

        assert spy.called


class TestEMAModel:
    """Tests for the EMAModel wrapper."""

    @pytest.fixture(scope='class')
    def ema_model(self) -> EMAModel[torch.Tensor, torch.Tensor]:
        """Fixture for the EMA model."""
        return EMAModel(ComplexModule(), name='ema_model', decay=0.9)

    def test_init(self, ema_model) -> None:
        """Test class initialization."""
        fn = ema_model._get_multi_avg_fn()
        assert callable(fn)
        assert ema_model.decay == 0.9

    def test_ema_update_parameters_changes_weights(self, ema_model) -> None:
        """EMA update should modify averaged weights."""
        initial = [p.clone() for p in ema_model.averaged_module.parameters()]
        for p in ema_model.module.parameters():
            p.data.zero_()

        ema_model.post_batch_update()

        updated = list(ema_model.averaged_module.parameters())
        assert any(
            not torch.allclose(p0, p1)
            for p0, p1 in zip(initial, updated, strict=False)
        )


class TestModelOptimizer:
    """Tests for ModelOptimizer."""

    @pytest.fixture(scope='class')
    def complex_model(self) -> Model[torch.Tensor, torch.Tensor]:
        """Fixture of a complex model wrapped with Model."""
        return Model(ComplexModule(), name='complex_model')

    @pytest.fixture()
    def model_optimizer(
        self, complex_model, mock_learning_schema
    ) -> ModelOptimizer:
        """Set up a test instance."""
        return ModelOptimizer(
            model=complex_model,
            learning_schema=mock_learning_schema,
        )

    def test_update_learning_rate_global(self, model_optimizer) -> None:
        """Test it correctly updates global learning rate."""
        model_optimizer.update_learning_rate(base_lr=0.02)

        for param_group in model_optimizer._optimizer.param_groups:
            assert param_group['lr'] == 0.02

    def test_update_learning_rate_param(self, model_optimizer) -> None:
        """Test it correctly updates parameter-specific learning rates."""
        dict_lr: dict[str, float] = {'linear': 0.01, 'linear2': 0.001}
        model_optimizer.update_learning_rate(base_lr=dict_lr)

        param_groups = model_optimizer._optimizer.param_groups
        for param_group, lr in zip(
            param_groups, dict_lr.values(), strict=False
        ):
            assert param_group['lr'] == lr

    def test_missing_param_error(self, model_optimizer) -> None:
        """Test that MissingParamError is raised when params are missing."""
        with pytest.raises(exceptions.MissingParamError):
            model_optimizer.base_lr = {'linear': 0.1}

"""Tests for the "models" module."""

import torch

import pytest

from drytorch import Model
from drytorch.core import exceptions
from drytorch.lib.models import ModelOptimizer


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

    def test_model_increment_epoch(self, complex_model: Model) -> None:
        """Test Model's increment_epoch method increases the epoch count."""
        complex_model.increment_epoch()
        assert complex_model.epoch == 1


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

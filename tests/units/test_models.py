"""Tests for the "models" module."""

import torch

import pytest

from drytorch import Model, exceptions
from drytorch.models import ModelOptimizer


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


class TestModelOptimizerGlobalLR:
    """Tests for ModelOptimizer with a global learning rate."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mock_experiment, mock_learning_scheme) -> None:
        """Set up a mock ModelOptimizer instance with global lr."""
        mock_model.module = ComplexModule()
        mock_learning_scheme.scheduler = lambda x, y: x
        self.model_optimizer = ModelOptimizer(
            model=mock_model,
            learning_scheme=mock_learning_scheme,
        )

    def test_update_learning_rate(self) -> None:
        """Test it correctly updates global learning rate."""
        self.model_optimizer.update_learning_rate(base_lr=0.02)

        for param_group in self.model_optimizer._optimizer.param_groups:
            assert param_group['lr'] == 0.02


class TestModelOptimizerParameterLR:
    """Tests for ModelOptimizer with parameter-specific learning rates."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mock_experiment, mock_learning_scheme) -> None:
        """Set up a ModelOptimizer instance with per-layer learning rates."""
        mock_model.module = ComplexModule()
        mock_learning_scheme.scheduler = lambda x, y: x
        self.dict_lr: dict[str, float] = {'linear': 0.01, 'linear2': 0.001}
        self.model_optimizer = ModelOptimizer(
            mock_model,
            learning_scheme=mock_learning_scheme,
        )

    def test_update_learning_rate(self) -> None:
        """Test it correctly updates parameter-specific learning rates."""
        new_lr = {key: value / 2 for key, value in self.dict_lr.items()}
        self.model_optimizer.update_learning_rate(base_lr=new_lr)

        param_groups = self.model_optimizer._optimizer.param_groups
        for param_group, lr in zip(param_groups, new_lr.values(), strict=False):
            assert param_group['lr'] == lr

    def test_missing_param_error(self) -> None:
        """Test that MissingParamError is raised when params are missing."""
        with pytest.raises(exceptions.MissingParamError):
            self.model_optimizer._base_lr = {'linear': 0.1}

"""Tests for the "learning" module."""

import pytest

import torch

from dry_torch import exceptions
from dry_torch.learning import LearningScheme, Model, ModelOptimizer


class ComplexModule(torch.nn.Module):
    """Example for an arbitrarily complex module."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(2, 1),
                                           torch.nn.Linear(1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard forward."""
        return self.linear2(self.relu(self.linear(inputs)))


class TestModel:

    @pytest.fixture(scope='class')
    def complex_model(self) -> Model[torch.Tensor, torch.Tensor]:
        """Fixture of a complex model."""
        complex_model = Model(ComplexModule(), name='complex_model')
        return complex_model

    def test_model_increment_epoch(self, complex_model: Model) -> None:
        """Test Model's increment_epoch method increases the epoch count."""
        complex_model.increment_epoch()
        assert complex_model.epoch == 1


class TestModelOptimizerGlobalLR:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mock_experiment) -> None:
        """Set up a mock ModelOptimizer instance with a global lr."""
        mock_model.module = ComplexModule()
        learning_scheme = LearningScheme(optimizer_cls=torch.optim.SGD,
                                         base_lr=0.01)
        self.model_optimizer = ModelOptimizer(model=mock_model,
                                              learning_scheme=learning_scheme)
        return

    def test_get_scheduled_lr(self, ) -> None:
        """Test get_scheduled_lr returns correct learning rates."""
        scheduled_lr = self.model_optimizer.get_opt_params()
        assert isinstance(scheduled_lr, list)
        assert len(scheduled_lr) == 1
        assert 'lr' in scheduled_lr[0]
        assert scheduled_lr[0]['lr'] == 0.01

    def test_update_learning_rate(self) -> None:
        """Test it correctly updates learning rates."""
        # Set new learning rate and check scheduler is disabled
        self.model_optimizer.update_learning_rate(base_lr=0.02)

        # Check optimizer parameter group learning rate
        for param_group in self.model_optimizer.optimizer.param_groups:
            assert param_group['lr'] == 0.02


class TestModelOptimizerParameterLR:
    @pytest.fixture(autouse=True)
    def setup(self, mock_model, mock_experiment) -> None:
        """Set up a mock ModelOptimizer instance with a lr per parameter."""
        mock_model.module = ComplexModule()
        self.dict_lr: dict[str, float] = {'linear': 0.01, 'linear2': 0.001}
        learning_scheme = LearningScheme(optimizer_cls=torch.optim.SGD,
                                         base_lr=self.dict_lr)
        self.model_optimizer = ModelOptimizer(mock_model,
                                              learning_scheme=learning_scheme)
        return

    def test_update_learning_rate(self) -> None:
        """Test it correctly updates learning rates."""
        new_lr = {key: value / 2 for key, value in self.dict_lr.items()}

        self.model_optimizer.update_learning_rate(base_lr=new_lr)

        param_groups = self.model_optimizer.optimizer.param_groups
        for param_group, lr in zip(param_groups, new_lr.values()):
            assert param_group['lr'] == lr

    def test_missing_param_error(self) -> None:
        """Test that MissingParamError is raised when params are missing."""

        with pytest.raises(exceptions.MissingParamError):
            self.model_optimizer.base_lr = {'linear': 0.1}

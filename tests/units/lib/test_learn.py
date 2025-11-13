"""Tests for the "learn" module."""

import dataclasses

import torch

import pytest

from drytorch.core import protocols as p
from drytorch.lib import schedulers
from drytorch.lib.learn import LearningSchema


@pytest.fixture
def mock_scheduler(mocker) -> p.SchedulerProtocol:
    """Fixture for a mock scheduler."""
    mock = mocker.create_autospec(
        spec=p.SchedulerProtocol, instance=True, side_effect=lambda x, y: x
    )
    return mock


class TestLearningScheme:
    """Tests for the LearningScheme class."""

    def test_is_dataclass(self):
        """Test if LearningScheme is a dataclass."""
        assert dataclasses.is_dataclass(LearningSchema)

    def test_fields(self):
        """Test the fields of the LearningScheme dataclass."""
        fields = dataclasses.fields(LearningSchema)
        assert len(fields) == 5
        field_names = [f.name for f in fields]
        assert 'optimizer_cls' in field_names
        assert 'base_lr' in field_names
        assert 'scheduler' in field_names
        assert 'optimizer_defaults' in field_names
        assert 'gradient_op' in field_names

    def test_default_values(self):
        """Test default values of a LearningScheme instance."""
        scheme = LearningSchema(optimizer_cls=torch.optim.Adam, base_lr=1e-3)
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)
        assert scheme.optimizer_defaults == {}
        assert scheme.gradient_op is None

    def test_adam(self, mock_scheduler):
        """Test the adam constructor method."""
        scheme = LearningSchema.adam()
        assert scheme.optimizer_cls == torch.optim.Adam
        assert scheme.base_lr == 1e-3
        assert scheme.optimizer_defaults == {'betas': (0.9, 0.999)}
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)

        scheme = LearningSchema.adam(
            base_lr=1e-2,
            betas=(0.8, 0.9),
            scheduler=mock_scheduler,
        )
        assert scheme.base_lr == 1e-2
        assert scheme.optimizer_defaults == {'betas': (0.8, 0.9)}
        assert scheme.scheduler == mock_scheduler

    def test_adam_w(self, mock_scheduler):
        """Test the adam_w constructor method."""
        scheme = LearningSchema.adam_w()
        assert scheme.optimizer_cls == torch.optim.AdamW
        assert scheme.base_lr == 1e-3
        assert scheme.optimizer_defaults == {
            'betas': (0.9, 0.999),
            'weight_decay': 1e-2,
        }
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)

        scheme = LearningSchema.adam_w(
            base_lr=1e-2,
            betas=(0.8, 0.9),
            weight_decay=1e-3,
            scheduler=mock_scheduler,
        )
        assert scheme.base_lr == 1e-2
        assert scheme.optimizer_defaults == {
            'betas': (0.8, 0.9),
            'weight_decay': 1e-3,
        }
        assert scheme.scheduler == mock_scheduler

    def test_sgd(self, mock_scheduler):
        """Test the sgd constructor method."""
        scheme = LearningSchema.sgd()
        assert scheme.optimizer_cls == torch.optim.SGD
        assert scheme.base_lr == 0.01
        assert scheme.optimizer_defaults == {
            'momentum': 0.0,
            'weight_decay': 0.0,
            'dampening': 0.0,
            'nesterov': False,
        }
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)

        scheme = LearningSchema.sgd(
            base_lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            dampening=0.1,
            nesterov=True,
            scheduler=mock_scheduler,
        )
        assert scheme.base_lr == 0.1
        assert scheme.optimizer_defaults == {
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'dampening': 0.1,
            'nesterov': True,
        }
        assert scheme.scheduler == mock_scheduler

    def test_r_adam(self, mock_scheduler):
        """Test the r_adam constructor method."""
        scheme = LearningSchema.r_adam()
        assert scheme.optimizer_cls == torch.optim.RAdam
        assert scheme.base_lr == 1e-3
        assert scheme.optimizer_defaults == {
            'betas': (0.9, 0.999),
            'weight_decay': 0.0,
            'decoupled_weight_decay': False,
        }
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)

        scheme = LearningSchema.r_adam(
            base_lr=1e-2,
            betas=(0.8, 0.9),
            weight_decay=1e-4,
            scheduler=mock_scheduler,
        )
        assert scheme.base_lr == 1e-2
        assert scheme.optimizer_defaults == {
            'betas': (0.8, 0.9),
            'weight_decay': 1e-4,
            'decoupled_weight_decay': True,
        }
        assert scheme.scheduler == mock_scheduler

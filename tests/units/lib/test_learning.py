"""Tests for the ``learning`` module."""

import dataclasses

from collections.abc import Mapping
from typing import Any

import torch

import pytest

from drytorch.core import protocols as p
from drytorch.lib import gradient_ops, schedulers
from drytorch.lib.learning import LearningSchema


@pytest.fixture
def mock_scheduler(mocker) -> p.SchedulerProtocol:
    """Fixture for a mock scheduler."""
    return mocker.create_autospec(
        spec=p.SchedulerProtocol, instance=True, side_effect=lambda x, y: x
    )


class TestLearningScheme:
    """Tests for the LearningSchema class."""

    def test_is_dataclass(self) -> None:
        """Test if LearningSchema is a dataclass."""
        assert dataclasses.is_dataclass(LearningSchema)

    def test_fields(self) -> None:
        """Test the fields of the LearningSchema dataclass."""
        fields = dataclasses.fields(LearningSchema)
        field_names = [f.name for f in fields]

        assert len(fields) == 5
        assert 'optimizer_cls' in field_names
        assert 'base_lr' in field_names
        assert 'scheduler' in field_names
        assert 'optimizer_defaults' in field_names
        assert 'gradient_op' in field_names

    def test_default_values(self) -> None:
        """Test default values of a LearningSchema instance."""
        scheme = LearningSchema(optimizer_cls=torch.optim.Adam, base_lr=1e-3)

        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)
        assert scheme.optimizer_defaults == {}
        assert isinstance(scheme.gradient_op, gradient_ops.NoOp)

    def test_adam_defaults(self) -> None:
        """Test the adam constructor method with defaults."""
        scheme = LearningSchema.adam()

        assert scheme.optimizer_cls == torch.optim.Adam
        assert scheme.base_lr == 1e-3
        assert scheme.optimizer_defaults == {'betas': (0.9, 0.999)}
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)

    def test_adam_custom(self, mock_scheduler) -> None:
        """Test the adam constructor method with custom parameters."""
        scheme = LearningSchema.adam(
            base_lr=1e-2,
            betas=(0.8, 0.9),
            scheduler=mock_scheduler,
        )

        assert scheme.base_lr == 1e-2
        assert scheme.optimizer_defaults == {'betas': (0.8, 0.9)}
        assert scheme.scheduler == mock_scheduler

    def test_adam_w_defaults(self) -> None:
        """Test the adam_w constructor method with defaults."""
        scheme = LearningSchema.adam_w()

        assert scheme.optimizer_cls == torch.optim.AdamW
        assert scheme.base_lr == 1e-3
        assert scheme.optimizer_defaults == {
            'betas': (0.9, 0.999),
            'weight_decay': 1e-2,
        }
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)

    def test_adam_w_custom(self, mock_scheduler) -> None:
        """Test the adam_w constructor method with custom parameters."""
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

    def test_sgd_defaults(self) -> None:
        """Test the sgd constructor method with defaults."""
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

    def test_sgd_custom(self, mock_scheduler) -> None:
        """Test the sgd constructor method with custom parameters."""
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

    def test_r_adam_defaults(self) -> None:
        """Test the r_adam constructor method with defaults."""
        scheme = LearningSchema.r_adam()

        assert scheme.optimizer_cls == torch.optim.RAdam
        assert scheme.base_lr == 1e-3
        assert scheme.optimizer_defaults == {
            'betas': (0.9, 0.999),
            'weight_decay': 0.0,
            'decoupled_weight_decay': False,
        }
        assert isinstance(scheme.scheduler, schedulers.ConstantScheduler)

    def test_r_adam_custom(self, mock_scheduler) -> None:
        """Test the r_adam constructor method with custom parameters."""
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

    def test_mapping_containers(self) -> None:
        """Test that base_lr and optimizer_defaults accept Mapping."""
        base_lr: Mapping[str, float] = {'linear': 0.01}
        optimizer_defaults: Mapping[str, Any] = {'weight_decay': 1e-4}

        scheme = LearningSchema(
            optimizer_cls=torch.optim.Adam,
            base_lr=base_lr,
            optimizer_defaults=optimizer_defaults,
        )

        assert scheme.base_lr == {'linear': 0.01}
        assert scheme.optimizer_defaults == {'weight_decay': 1e-4}

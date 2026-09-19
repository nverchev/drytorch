"""Tests that the library classes match the protocols.

basedpyright checks the assignments below: one fails when a class does not
match the protocol. The tests check the protocols that are runtime-checkable.
"""

from collections.abc import Generator
from typing import Any

import pytest

from drytorch.core import protocols as p
from drytorch.lib.checkpoints import LocalCheckpoint
from drytorch.lib.evaluations import Diagnostic, Validation
from drytorch.lib.evaluations import Test as _Test
from drytorch.lib.gradient_ops import HistClipper, NoOp
from drytorch.lib.learning import LearningSchema
from drytorch.lib.loading import DataLoader
from drytorch.lib.models import EMAModel, Model, SWAModel
from drytorch.lib.objectives import (
    DictLoss,
    JoinedLoss,
    JoinedMetrics,
    Loss,
    LossWithMetrics,
    Metric,
    MetricCollection,
)
from drytorch.lib.runners import Runner, Source
from drytorch.lib.schedulers import ConstantScheduler, ExponentialScheduler
from drytorch.lib.training import Trainer


_runner: type[p.RunnerProtocol] = Runner
_checkpoint: type[p.CheckpointProtocol] = LocalCheckpoint
_clipper: type[p.GradientOpProtocol] = HistClipper
_diagnostic: type[p.SourceProtocol] = Diagnostic
_dict_loss: type[p.LossProtocol[Any, Any]] = DictLoss
_ema_model: type[p.ModelProtocol[Any, Any]] = EMAModel
_gradient_op: type[p.GradientOpProtocol] = NoOp
_higher_is_better: type[p.HigherIsBetterProtocol] = Metric
_joined_loss: type[p.LossProtocol[Any, Any]] = JoinedLoss
_joined_metrics: type[p.ObjectiveProtocol[Any, Any]] = JoinedMetrics
_learning_schema: type[p.LearningProtocol] = LearningSchema
_loader: type[p.LoaderProtocol[Any]] = DataLoader
_loss: type[p.LossProtocol[Any, Any]] = Loss
_loss_with_metrics: type[p.LossProtocol[Any, Any]] = LossWithMetrics
_metric: type[p.ObjectiveProtocol[Any, Any]] = Metric
_metrics: type[p.ObjectiveProtocol[Any, Any]] = MetricCollection
_model: type[p.ModelProtocol[Any, Any]] = Model
_named: type[p.NamedProtocol] = Model
_source: type[p.SourceProtocol] = Source
_scheduler: type[p.SchedulerProtocol] = ConstantScheduler
_swa_model: type[p.ModelProtocol[Any, Any]] = SWAModel
_sync: type[p.SyncProtocol] = MetricCollection
_test: type[p.TestProtocol] = _Test
_trainer: type[p.TrainerProtocol[Any, Any, Any]] = Trainer
_transformed: type[p.SchedulerProtocol] = ExponentialScheduler
_validation: type[p.ValidationProtocol] = Validation


@pytest.fixture(autouse=True, scope='module')
def start_experiment(run) -> Generator[None, None, None]:
    """Create an experimental scope for the tests."""
    yield
    return


@pytest.mark.parametrize(
    ('fixture_name', 'protocol'),
    [
        ('identity_trainer', p.RunnerProtocol),
        ('identity_trainer', p.NamedProtocol),
        ('identity_trainer', p.TrainerProtocol),
        ('linear_model', p.ModelProtocol),
        ('linear_model', p.NamedProtocol),
        ('square_loss_calc', p.HigherIsBetterProtocol),
        ('square_loss_calc', p.LossProtocol),
        ('square_loss_calc', p.NamedProtocol),
        ('square_loss_calc', p.SyncProtocol),
        ('zero_metrics_calc', p.HigherIsBetterProtocol),
        ('zero_metrics_calc', p.ObjectiveProtocol),
    ],
)
def test_library_object_matches_protocol(
    request, fixture_name, protocol
) -> None:
    """Test the object the fixture provides satisfies the protocol."""
    instance = request.getfixturevalue(fixture_name)

    assert isinstance(instance, protocol)

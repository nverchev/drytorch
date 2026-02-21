"""Functional tests to check convergence of trainers."""

from collections.abc import Generator

import torch

import pytest

from drytorch.core.experiment import Run


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


def test_convergence(linear_model, identity_trainer) -> None:
    """Test trainer convergence to 1."""
    identity_trainer.train(10)
    linear_weight = next(linear_model.module.parameters())
    assert torch.isclose(linear_weight, torch.tensor(1.0), atol=0.1)


def test_swa_convergence(swa_model, identity_trainer_with_swa) -> None:
    """Test trainer convergence to 1."""
    identity_trainer_with_swa.train(10)
    base_weight = next(swa_model.module.parameters())
    avg_weight = next(swa_model.averaged_module.parameters())

    # close to optimal but not identical
    assert torch.isclose(base_weight, torch.tensor(1.0), atol=0.1)
    assert torch.isclose(avg_weight, torch.tensor(1.0), atol=0.1)
    assert not torch.isclose(base_weight, avg_weight)


def test_ema_convergence(ema_model, identity_trainer_with_ema) -> None:
    """Test trainer convergence to 1."""
    identity_trainer_with_ema.train(5)
    base_weight_halfway = next(ema_model.module.parameters()).clone()
    avg_weight_halfway = next(ema_model.averaged_module.parameters()).clone()

    identity_trainer_with_ema.train(5)
    base_weight_final = next(ema_model.module.parameters())
    avg_weight_final = next(ema_model.averaged_module.parameters())

    # slower convergence than the training model
    assert not torch.isclose(avg_weight_halfway, base_weight_halfway)
    assert torch.isclose(base_weight_final, torch.tensor(1.0), atol=0.1)
    assert torch.isclose(avg_weight_final, torch.tensor(1.0), atol=0.1)


def test_ema_slow_convergence(ema_model, identity_trainer_with_ema) -> None:
    """Test trainer convergence to 1."""
    ema_model.decay = 0.999
    ema_model.averaged_module = ema_model._create_averaged_module()

    identity_trainer_with_ema.train(10)
    avg_weight = next(ema_model.averaged_module.parameters())

    assert not torch.isclose(avg_weight, torch.tensor(1.0), atol=0.1)

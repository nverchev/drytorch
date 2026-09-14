"""Integration tests for registering models inside and across runs."""

import pathlib

import torch

import pytest

import drytorch

from drytorch.core import exceptions, registering
from drytorch.core.experimenting import Experiment
from drytorch.lib.checkpoints import LocalCheckpoint
from drytorch.lib.models import Model


class _Actor:
    """Object acting on a model."""

    def __init__(self, model: Model | None = None) -> None:
        if model is not None:
            self.model = model


class TestModelRegistration:
    """Test model registration outside, across, and after runs."""

    @pytest.fixture
    def experiment(self, tmp_path) -> Experiment[None]:
        """Experiment without default trackers in a temporary folder."""
        drytorch.remove_all_default_trackers()
        return Experiment(config=None, name='RegistrationExp', par_dir=tmp_path)

    def test_model_outside_run(self) -> None:
        """Test Model raises NoActiveExperimentError outside a run."""
        module = torch.nn.Linear(1, 1)

        with pytest.raises(exceptions.NoActiveExperimentError):
            Model(module)

    def test_actor_after_resume(self, experiment) -> None:
        """Test a model registered in a run is accepted after resuming it."""
        actor = _Actor()
        with experiment.create_run():
            model = Model(torch.nn.Linear(1, 1))

        with experiment.create_run(resume=True):
            registering.register_actor(actor, model)

        assert actor in registering.ALL_ACTORS[model.module]

    def test_actor_in_new_run(self, experiment) -> None:
        """Test a model from a previous run is rejected in a new run."""
        with experiment.create_run():
            model = Model(torch.nn.Linear(1, 1))

        with experiment.create_run():
            with pytest.raises(exceptions.ModuleFromAnotherRunError):
                registering.register_actor(_Actor(), model)

    def test_actor_records_checkpoint_par_dir(
        self, experiment, tmp_path: pathlib.Path
    ) -> None:
        """Test actor with model records custom checkpoint par_dir."""
        custom_par_dir = tmp_path / 'weights'
        checkpoint = LocalCheckpoint(custom_par_dir)

        with experiment.create_run() as run:
            model = Model(torch.nn.Linear(1, 1), checkpoint=checkpoint)
            actor = _Actor(model)
            registering.register_actor(actor, model)
            metadata = run.metadata_manager.named_metadata['_Actor']

        assert isinstance(metadata, dict)
        assert metadata['model']['checkpoint'] == {
            'class': 'LocalCheckpoint',
            'par_dir': repr(custom_par_dir),
        }

    def test_actor_records_default_checkpoint(self, experiment) -> None:
        """Test actor with default checkpoint records class name."""
        with experiment.create_run() as run:
            model = Model(torch.nn.Linear(1, 1))
            actor = _Actor(model)
            registering.register_actor(actor, model)
            metadata = run.metadata_manager.named_metadata['_Actor']

        assert isinstance(metadata, dict)
        assert metadata['model']['checkpoint'] == 'LocalCheckpoint'

"""Integration tests for registering models inside and across runs."""

import pathlib

import torch

import pytest

import drytorch

from drytorch.core import exceptions, registering
from drytorch.core.experimenting import Experiment
from drytorch.lib.checkpoints import LocalCheckpoint
from drytorch.lib.loading import DataLoader
from drytorch.lib.models import Model
from drytorch.lib.runners import Runner


class _Registered:
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
        """Test ``Model`` raises ``NoActiveExperimentError`` outside a run."""
        module = torch.nn.Linear(1, 1)

        with pytest.raises(exceptions.NoActiveExperimentError):
            Model(module)

    def test_register_runner_after_resume(self, experiment) -> None:
        """Test a model registered in a run is accepted after resuming it."""
        runner = _Registered()
        with experiment.create_run():
            model = Model(torch.nn.Linear(1, 1))

        with experiment.create_run(resume=True):
            registering.register_runner(runner, model)

        assert runner in registering.ALL_RUNNERS[model.module]

    def test_register_runner_in_new_run(self, experiment) -> None:
        """Test a model from a previous run is rejected in a new run."""
        with experiment.create_run():
            model = Model(torch.nn.Linear(1, 1))

        with experiment.create_run():
            with pytest.raises(exceptions.ModuleFromAnotherRunError):
                registering.register_runner(_Registered(), model)

    def test_runner_records_checkpoint_par_dir(
        self, experiment, tmp_path: pathlib.Path
    ) -> None:
        """Test runner with model records custom checkpoint par_dir."""
        custom_par_dir = tmp_path / 'weights'
        checkpoint = LocalCheckpoint(custom_par_dir)

        with experiment.create_run() as run:
            model = Model(torch.nn.Linear(1, 1), checkpoint=checkpoint)
            runner = _Registered(model)
            registering.register_runner(runner, model)
            metadata = run.metadata_manager.named_metadata['_Registered']

        assert isinstance(metadata, dict)
        assert metadata['model']['checkpoint'] == {
            'class': 'LocalCheckpoint',
            'par_dir': repr(custom_par_dir),
        }

    def test_runner_records_default_checkpoint(self, experiment) -> None:
        """Test runner with default checkpoint records class name."""
        with experiment.create_run() as run:
            model = Model(torch.nn.Linear(1, 1))
            runner = _Registered(model)
            registering.register_runner(runner, model)
            metadata = run.metadata_manager.named_metadata['_Registered']

        assert isinstance(metadata, dict)
        assert metadata['model']['checkpoint'] == 'LocalCheckpoint'

    def test_runner_registers_when_created(self, experiment) -> None:
        """Test a runner registers as a runner of its model when created."""
        dataset = torch.utils.data.TensorDataset(torch.ones(4, 1))
        with experiment.create_run():
            model = Model(torch.nn.Linear(1, 1))
            runner = Runner(model, loader=DataLoader(dataset, 2))

        assert runner in registering.ALL_RUNNERS[model.module]

    def test_runner_in_new_run(self, experiment) -> None:
        """Test a runner is rejected for a model from a previous run."""
        dataset = torch.utils.data.TensorDataset(torch.ones(4, 1))
        with experiment.create_run():
            model = Model(torch.nn.Linear(1, 1))

        with experiment.create_run():
            with pytest.raises(exceptions.ModuleFromAnotherRunError):
                Runner(model, loader=DataLoader(dataset, 2))

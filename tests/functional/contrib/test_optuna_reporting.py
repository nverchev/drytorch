"""Test reporting on optuna trial's performance."""

from collections.abc import Generator

import optuna
import torch

import pytest

from drytorch.contrib.optuna import TrialCallback, get_final_value
from drytorch.core.experiment import Run
from tests.simple_classes import TorchData


@pytest.fixture(autouse=True, scope='module')
def autorun_experiment(run) -> Generator[Run, None, None]:
    """Create an experimental scope for the tests."""
    yield run
    return


class TestTrialCallbackAsk:
    """Test report and retrial of optuna trial values when created with ask."""

    @pytest.fixture
    def trial(self) -> optuna.Trial:
        """Create a TrialCallback instance for testing."""
        study = optuna.create_study(direction='minimize')
        return study.ask()

    @pytest.fixture
    def trial_callback(self, trial) -> TrialCallback:
        """Create a TrialCallback instance for testing."""
        return TrialCallback(trial=trial, best_is='lower')

    def test_reported(self, trial_callback, identity_trainer) -> None:
        """Test reported values."""
        identity_trainer.model.epoch = 3
        identity_trainer.objective.update(
            TorchData(torch.ones(2)), torch.zeros(2)
        )
        trial_callback(identity_trainer)
        # here you can tell a result with trial.study.tell(trial, {your_result})
        assert trial_callback.reported == {3: 1}


class TestTrialCallbackObjective:
    """Test report and retrial of optuna trial values with optimized syntax."""

    @pytest.fixture
    def study(self) -> optuna.Study:
        """Create a TrialCallback instance for testing."""
        return optuna.create_study(direction='minimize')

    @pytest.fixture(autouse=True)
    def optimize(self, study, identity_trainer) -> None:
        """Create a TrialCallback instance for testing."""

        def _objective(trial: optuna.Trial) -> float:
            identity_trainer.model.epoch = 3
            identity_trainer.objective.update(
                TorchData(torch.ones(2)), torch.zeros(2)
            )
            trial_callback = TrialCallback(trial=trial, best_is='lower')
            trial_callback(identity_trainer)
            return get_final_value(trial)

        study.optimize(_objective, n_trials=1)
        return

    def test_get_objective_value(self, study) -> None:
        """Test get_objective_value return the correct value."""
        assert study.trials[-1].value == 1

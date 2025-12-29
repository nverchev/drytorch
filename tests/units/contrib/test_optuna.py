"""Tests for the "optuna" module."""

import pytest


try:
    import optuna

    from omegaconf import OmegaConf
except ImportError:
    pytest.skip('optuna not available', allow_module_level=True)
    raise

from drytorch.contrib import optuna as dry_optuna


class TestTrialCallback:
    """Tests for the TrialCallback class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_monitor = mocker.patch('drytorch.lib.hooks.MetricMonitor')
        self.mock_monitor_instance = self.mock_monitor.return_value
        self.mock_monitor_instance.filtered_value = 0.5
        self.mock_monitor_instance.metric_name = 'val_loss'
        self.mock_trial = mocker.Mock(spec=optuna.Trial)
        return

    @pytest.fixture
    def mock_trial(self, mocker) -> optuna.Trial:
        """Mock trial object."""
        return mocker.Mock(spec=optuna.Trial)

    @pytest.fixture
    def callback(
        self, mock_trial, example_loss_name
    ) -> dry_optuna.TrialCallback:
        """Mock callback."""
        return dry_optuna.TrialCallback(mock_trial, metric=example_loss_name)

    def test_call_reports_to_trial(
        self, mock_trial, callback, mock_trainer, example_epoch
    ) -> None:
        """Test that the callback reports the value to the trial."""
        mock_trial.should_prune.return_value = False
        callback(mock_trainer)
        self.mock_monitor_instance.record_metric_value.assert_called_once_with(
            mock_trainer
        )
        mock_trial.report.assert_called_once_with(0.5, example_epoch)
        assert callback.reported == {example_epoch: 0.5}

    def test_pruning(self, mock_trial, callback, mock_trainer) -> None:
        """Test the callback raises TrialPruned if the trial should prune."""
        mock_trial.should_prune.return_value = True
        with pytest.raises(optuna.TrialPruned):
            callback(mock_trainer)

        call_argument = mock_trainer.terminate_training.call_args[0][0]

        mock_trainer.terminate_training.assert_called_once()
        assert isinstance(call_argument, str)
        assert call_argument.startswith('Optuna pruning')


class TestSuggestOverrides:
    """Tests for suggest_overrides function."""

    @pytest.fixture
    def mock_trial(self, mocker):
        """Mock trial object."""
        return mocker.Mock(spec=optuna.Trial)

    @pytest.fixture
    def basic_float_config(self):
        """Config with a single float parameter."""
        return OmegaConf.create(
            {
                'tune': {
                    'params': {
                        'model.lr': {
                            'suggest': 'suggest_float',
                            'settings': {'low': 0.001, 'high': 0.1},
                        }
                    }
                },
                'overrides': ['experiment=test'],
            }
        )

    @pytest.fixture
    def nested_list_config(self):
        """Config with an integer parameters list of variable length."""
        return OmegaConf.create(
            {
                'tune': {
                    'params': {
                        'layers': {
                            'suggest': 'suggest_list',
                            'settings': {
                                'min_length': 1,
                                'max_length': 3,
                                'suggest': 'suggest_int',
                                'settings': {'low': 10, 'high': 20},
                            },
                        }
                    }
                },
                'overrides': [],
            }
        )

    @pytest.fixture
    def invalid_method_config(self):
        """Config with non-existent optuna method."""
        return OmegaConf.create(
            {
                'tune': {
                    'params': {
                        'param': {
                            'suggest': 'non_existent_method',
                            'settings': {},
                        }
                    }
                },
                'overrides': [],
            }
        )

    @pytest.fixture
    def invalid_list_method_config(self):
        """Config with a non-existent method in list settings."""
        return OmegaConf.create(
            {
                'tune': {
                    'params': {
                        'param': {
                            'suggest': 'suggest_list',
                            'settings': {
                                'min_length': 1,
                                'max_length': 2,
                                'suggest': 'non_existent_method',
                                'settings': {},
                            },
                        }
                    }
                },
                'overrides': [],
            }
        )

    def test_suggest_overrides_basic(self, mock_trial, basic_float_config):
        """Test basic scalar suggestion."""
        mock_trial.suggest_float.return_value = 0.05

        overrides = dry_optuna.suggest_overrides(basic_float_config, mock_trial)

        mock_trial.suggest_float.assert_called_with('Lr', low=0.001, high=0.1)
        assert 'experiment=test' in overrides
        assert 'model.lr=0.05' in overrides

    def test_suggest_overrides_full_name(self, mock_trial, basic_float_config):
        """Test suggestion with use_full_name=True."""
        mock_trial.suggest_float.return_value = 0.05

        overrides = dry_optuna.suggest_overrides(
            basic_float_config, mock_trial, use_full_name=True
        )

        mock_trial.suggest_float.assert_called_with(
            'model.lr', low=0.001, high=0.1
        )
        assert 'model.lr=0.05' in overrides

    def test_suggest_overrides_nested_list(
        self, mock_trial, nested_list_config
    ):
        """Test suggest_list logic."""
        # patch list length, first value, second value
        mock_trial.suggest_int.side_effect = [2, 15, 18]

        overrides = dry_optuna.suggest_overrides(nested_list_config, mock_trial)

        mock_trial.suggest_int.assert_any_call(
            name='Num. Layers', low=1, high=3
        )
        mock_trial.suggest_int.assert_any_call('Layers 0', low=10, high=20)
        mock_trial.suggest_int.assert_any_call('Layers 1', low=10, high=20)

        assert 'layers=[15, 18]' in overrides

    def test_invalid_config_raises(self, mock_trial, invalid_method_config):
        """Test invalid optuna method raises DryTorchError."""
        with pytest.raises(dry_optuna.OptunaError, match='configuration'):
            dry_optuna.suggest_overrides(invalid_method_config, mock_trial)

    def test_invalid_list_config_raises(
        self, mock_trial, invalid_list_method_config
    ):
        """Test invalid optuna method in a list config raises DryTorchError."""
        mock_trial.suggest_int.return_value = 1

        with pytest.raises(dry_optuna.OptunaError, match='configuration'):
            dry_optuna.suggest_overrides(invalid_list_method_config, mock_trial)


@pytest.fixture
def mock_trial_intermediate_values(mocker):
    """Mock trial with intermediate values."""
    trial = mocker.Mock(spec=optuna.Trial)
    trial.number = 0
    study = mocker.Mock()
    study.direction.name = 'MINIMIZE'
    frozen = mocker.Mock()
    frozen.number = 0
    frozen.intermediate_values = {1: 0.8, 2: 0.5, 3: 0.6}
    study.trials = [frozen]
    trial.study = study
    return trial


def test_get_final_value_basic(mock_trial_intermediate_values):
    """Test basic retrieval."""
    val = dry_optuna.get_final_value(mock_trial_intermediate_values)
    assert val == 0.5


def test_trial_number_mismatch(mock_trial_intermediate_values):
    """Test error on trial mismatch."""
    mock_trial_intermediate_values.study.trials[-1].number = 999
    with pytest.raises(dry_optuna.OptunaError, match='trial number mismatch'):
        dry_optuna.get_final_value(mock_trial_intermediate_values)


def test_no_reported_values(mock_trial_intermediate_values):
    """Test error when no values reported."""
    mock_trial_intermediate_values.study.trials[-1].intermediate_values = {}
    with pytest.raises(dry_optuna.OptunaError, match='has no reported values'):
        dry_optuna.get_final_value(mock_trial_intermediate_values)

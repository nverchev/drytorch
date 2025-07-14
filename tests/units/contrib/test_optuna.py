"""Unit tests for the "optuna" module."""

from collections.abc import Callable, Sequence

import pytest

try:
    import optuna
except ImportError:
    pytest.skip('optuna not available', allow_module_level=True)
    raise

try:
    from omegaconf import OmegaConf, DictConfig
except ImportError:
    pytest.skip('omegaconf', allow_module_level=True)
    raise

from drytorch import exceptions
from drytorch.contrib.optuna import TrialCallback
from drytorch.contrib.optuna import suggest_overrides
from drytorch.contrib.optuna import get_final_value


class TestTrialCallback:
    """Test cases for TrialCallback class."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up the tests."""
        self.mock_metric_monitor = mocker.Mock()
        self.mock_metric_monitor.filtered_value = 0.5
        self.mock_metric_monitor.metric_name = 'test_metric'
        self.mock_metric_monitor.record_metric_value = mocker.Mock()
        mock_monitor_class = mocker.patch('drytorch.hooks.MetricMonitor')
        mock_monitor_class.return_value = self.mock_metric_monitor

    @pytest.fixture
    def mock_trial(self, mocker) -> optuna.Trial:
        """Create a mock optuna trial."""
        trial = mocker.create_autospec(optuna.Trial)
        trial.report = mocker.Mock()
        trial.should_prune = mocker.Mock(return_value=False)
        return trial

    @pytest.fixture
    def mock_aggregate_fn(self, mocker) -> Callable[[Sequence[float]], float]:
        """Create a mock aggregate function."""
        return mocker.Mock(return_value=0.5)

    @pytest.fixture
    def trial_callback(self,
                       mock_trial,
                       mock_aggregate_fn) -> TrialCallback:
        """Create a TrialCallback instance for testing."""

        callback = TrialCallback(
            trial=mock_trial,
            filter_fn=mock_aggregate_fn,
            metric='test_metric',
            best_is='higher'
        )
        return callback

    def test_init_creates_metric_monitor(self,
                                         trial_callback,
                                         mock_trial,
                                         mock_aggregate_fn) -> None:
        """Test that TrialCallback properly initializes MetricMonitor."""

        assert trial_callback.trial == mock_trial
        assert trial_callback.monitor == self.mock_metric_monitor
        assert trial_callback.reported == {}

    def test_call_reports_to_trial(self,
                                   trial_callback,
                                   mock_trainer,
                                   mock_model,
                                   mock_trial) -> None:
        """Test that callback reports metrics to the trial."""
        trial_callback(mock_trainer)
        trial_callback.monitor.record_metric_value.assert_called_once_with(
            mock_trainer
        )
        # type: ignore
        mock_trial.report.assert_called_once_with(0.5,  # type: ignore
                                                  mock_model.epoch)
        assert trial_callback.reported == {3: 0.5}

    def test_call_no_pruning_when_should_prune_false(self,
                                                     trial_callback,
                                                     mock_trainer,
                                                     mock_trial) -> None:
        """Test that training continues when should_prune returns False."""
        mock_trial.should_prune.return_value = False

        trial_callback(mock_trainer)

        mock_trainer.terminate_training.assert_not_called()  # type: ignore

    def test_call_prunes_when_should_prune_true(self,
                                                trial_callback,
                                                mock_trainer,
                                                mock_trial) -> None:
        """Test that training is terminated when should_prune returns True."""
        mock_trial.should_prune.return_value = True

        with pytest.raises(optuna.TrialPruned):
            trial_callback(mock_trainer)

        mock_trainer.terminate_training.assert_called_once()  # type: ignore


class TestSuggestOverrides:
    """Test cases for the suggest_overrides function."""

    @pytest.fixture
    def mock_trial(self, mocker) -> optuna.Trial:
        """Create a mock optuna trial with suggestion methods."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_int = mocker.Mock(return_value=2)
        trial.suggest_float = mocker.Mock(return_value=0.01)
        trial.suggest_categorical = mocker.Mock(return_value='adam')
        return trial

    @pytest.fixture
    def basic_tune_cfg(self) -> DictConfig:
        """Create a basic tune configuration."""
        return OmegaConf.create({
            'overrides': ['model.name=ResNet'],
            'tune': {
                'params': {
                    'batch_size': {
                        'suggest': 'suggest_int',
                        'settings': {'low': 16, 'high': 128}
                    },
                    'optimizer': {
                        'suggest': 'suggest_categorical',
                        'settings': {'choices': ['adam', 'sgd']}
                    }
                }
            }
        })

    @pytest.fixture
    def list_tune_cfg(self) -> DictConfig:
        """Create a tune configuration with suggest_list."""
        return OmegaConf.create({
            'overrides': ['base.setting=true'],
            'tune': {
                'params': {
                    'dropouts': {
                        'suggest': 'suggest_list',
                        'settings': {
                            'min_length': 1,
                            'max_length': 3,
                            'suggest': 'suggest_float',
                            'settings': {'low': 0., 'high': 0.4}

                        }
                    }
                }
            }
        })

    @pytest.fixture
    def invalid_tune_cfg(self) -> DictConfig:
        """Create a basic tune configuration."""
        return OmegaConf.create({
            'overrides': [],
            'tune': {
                'params': {
                    'param1': {
                        'suggest': 'invalid_method',
                        'settings': {}
                    }
                }
            }
        })

    @pytest.fixture
    def invalid_list_tune_cfg(self) -> DictConfig:
        """Create a tune configuration with suggest_list."""
        return OmegaConf.create({
            'overrides': [],
            'tune': {
                'params': {
                    'param_list': {
                        'suggest': 'suggest_list',
                        'settings': {
                            'min_length': 1,
                            'max_length': 2,
                            'suggest': 'invalid_method',
                            'settings': {}
                        }
                    }
                }
            }
        })

    def test_suggest_overrides_basic_params(self,
                                            basic_tune_cfg,
                                            mock_trial) -> None:
        """Test suggest_overrides with basic parameters."""
        result = suggest_overrides(basic_tune_cfg, mock_trial)
        expected = [
            'model.name=ResNet',
            'batch_size=2',
            'optimizer=adam'
        ]
        assert result == expected

        mock_trial.suggest_int.assert_called_once_with(  # type: ignore
            'batch_size',
            low=16,
            high=128,
        )
        mock_trial.suggest_categorical.assert_called_once_with(  # type: ignore
            'optimizer',
            choices=['adam', 'sgd'],
        )

    def test_suggest_overrides_list_params(self,
                                           list_tune_cfg,
                                           mock_trial) -> None:
        """Test suggest_overrides with suggest_list parameters."""

        result = suggest_overrides(list_tune_cfg, mock_trial)

        expected = [
            'base.setting=true',
            'dropouts=[0.01, 0.01]'
        ]

        assert result == expected
        mock_trial.suggest_int.assert_called_once_with(  # type: ignore
            name='dropouts_len',
            low=1,
            high=3,
        )
        mock_trial.suggest_float.assert_any_call('dropouts_0',  # type: ignore
                                                 low=0,
                                                 high=0.4)
        mock_trial.suggest_float.assert_any_call('dropouts_1',  # type: ignore
                                                 low=0,
                                                 high=0.4)

    def test_suggest_overrides_empty_params(self,
                                            mock_trial) -> None:
        """Test suggest_overrides with empty parameters."""
        cfg = OmegaConf.create({
            'overrides': ['setting=value'],
            'tune': {'params': {}}
        })

        result = suggest_overrides(cfg, mock_trial)

        assert result == ['setting=value']

    def test_suggest_overrides_invalid_suggest_method(self,
                                                      invalid_tune_cfg,
                                                      mock_trial) -> None:
        """Test suggest_overrides with an invalid suggestion method."""
        with pytest.raises(exceptions.DryTorchException):
            suggest_overrides(invalid_tune_cfg, mock_trial)

    def test_invalid_list_suggest_method(self,
                                         invalid_list_tune_cfg,
                                         mock_trial) -> None:
        """Test invalid suggestion method in list settings."""
        with pytest.raises(exceptions.DryTorchException):
            suggest_overrides(invalid_list_tune_cfg, mock_trial)


class TestGetTrialValue:
    """Test cases for the get_trial_value function."""

    @pytest.fixture
    def mock_trial(self, mocker) -> optuna.Trial:
        """Set up the mock for the tests."""
        frozen_trial = mocker.create_autospec(optuna.trial.FrozenTrial,
                                              instance=True)
        frozen_trial.intermediate_values = {1: 2, 3: 4}
        frozen_trial.number = 0
        study = mocker.create_autospec(optuna.Study, instance=True)
        study.direction = mocker.Mock()
        study.direction.name = 'MINIMIZE'
        study.trials = [frozen_trial]
        trial = mocker.create_autospec(optuna.Trial, instance=True)
        trial.study = study
        trial.number = frozen_trial.number
        return trial

    def test_get_trial_value_minimize_with_default_fn(self, mock_trial) -> None:
        """Test get_trial_value with minimize direction and default function."""
        result = get_final_value(mock_trial)
        assert result == 2

    def test_get_trial_value_maximize_with_default_fn(self, mock_trial) -> None:
        """Test get_trial_value with maximize direction and default function."""
        mock_trial.study.direction.name = 'MAXIMISE'

        result = get_final_value(mock_trial)

        assert result == 4

    def test_get_trial_value_with_custom_average_fn(self, mock_trial) -> None:
        """Test get_trial_value with custom average function."""

        def _custom_avg(values):
            return sum(values) / len(values)

        result = get_final_value(mock_trial, filter_fn=_custom_avg)

        assert result == 3  # (2 + 4) / 2

    def test_get_trial_value_empty_intermediate_values(self,
                                                       mock_trial) -> None:
        """Test get_trial_value with empty intermediate values."""
        mock_trial.study.trials[0].intermediate_values = {}
        with pytest.raises(exceptions.DryTorchException):
            _ = get_final_value(mock_trial)


if __name__ == '__main__':
    pytest.main([__file__])

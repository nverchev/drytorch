"""Test optuna trial suggestions can be implemented as overrides."""

import pathlib

from collections.abc import Generator

import hydra
import optuna

from omegaconf import DictConfig, OmegaConf

import pytest

from drytorch.contrib.optuna import suggest_overrides


class TestSuggestOverridesWithHydra:
    """Integration tests for the optuna support module."""

    @pytest.fixture
    def trial(self) -> optuna.Trial:
        """A test instance."""
        study = optuna.create_study()
        return study.ask()

    @pytest.fixture
    def tune_cfg(self) -> DictConfig:
        """A test configuration."""
        return OmegaConf.create(
            {
                'overrides': [],
                'tune': {
                    'params': {
                        'model.type': {
                            'suggest': 'suggest_categorical',
                            'settings': {'choices': ['cnn', 'mlp']},
                        },
                        'batch_size': {
                            'suggest': 'suggest_int',
                            'settings': {'low': 16, 'high': 256, 'log': True},
                        },
                        'dropouts': {
                            'suggest': 'suggest_list',
                            'settings': {
                                'min_length': 1,
                                'max_length': 3,
                                'suggest': 'suggest_float',
                                'settings': {'low': 0.0, 'high': 0.5},
                            },
                        },
                    }
                },
            }
        )

    @pytest.fixture
    def config_name(self) -> Generator[str, None, None]:
        """The name of the overall experiment's configuration."""
        config_name = 'your_hydra_config'
        parent_dir = pathlib.Path(__file__).parent
        config_file = (parent_dir / config_name).with_suffix('.yaml')
        config_content = """
        model:
          type: mlp

        batch_size: 32

        dropouts: [0.1, 0.2]
        """
        with config_file.open('w') as f:
            f.write(config_content)

        yield config_name

        config_file.unlink()
        return

    def test_suggest_overrides_from_cfg(self, trial, tune_cfg) -> None:
        """Test suggest_overrides with actual optuna trial."""
        overrides = suggest_overrides(tune_cfg, trial)

        assert len(overrides) == 3
        assert overrides[0] in {'model.type=cnn', 'model.type=mlp'}
        assert overrides[1].startswith('batch_size=')
        assert overrides[2].startswith('dropouts=')

        # verify the values are within expected ranges
        batch_size = float(overrides[1].split('=')[1])
        dropout_values = overrides[2].split('=')[1].strip('[]').split(',')

        assert 16 <= batch_size <= 256
        for dropout in dropout_values:
            assert 0.0 <= float(dropout) <= 0.5

    def test_hydra_accepts_overrides(
        self, trial, tune_cfg, config_name
    ) -> None:
        """Test suggest_overrides with actual optuna trial."""
        with hydra.initialize(version_base=None, config_path='.'):
            overrides = suggest_overrides(tune_cfg, trial)
            dict_cfg = hydra.compose(
                config_name=config_name,
                overrides=overrides,
            )
        assert dict_cfg.model.type in {'cnn', 'mlp'}
        assert 16 <= dict_cfg.batch_size <= 256
        for dropout in dict_cfg.dropouts:
            assert 0.0 <= float(dropout) <= 0.5

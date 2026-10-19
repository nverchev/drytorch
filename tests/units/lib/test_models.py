"""Tests for the ``models`` module."""

import torch

import pytest

from drytorch.core import log_events
from drytorch.lib.models import EMAModel, Model, SWAModel, _get_auto_device
from tests.simple_classes import MLP, TorchData, TorchTuple


@pytest.fixture(autouse=True, scope='module')
def setup_module(module_mocker) -> None:
    """Fixture for a mock experiment."""
    module_mocker.patch('drytorch.core.registering.register_model')
    return


class TestModel:
    """Tests for the Model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up torch.autocast mocks."""
        self.mock_autocast = mocker.patch('torch.autocast')
        self.mock_context = mocker.Mock()
        self.mock_autocast.return_value.__enter__ = mocker.Mock(
            return_value=self.mock_context
        )
        self.mock_autocast.return_value.__exit__ = mocker.Mock(
            return_value=None
        )
        self.mock_compile = mocker.patch(
            'torch.compile', side_effect=lambda x: x
        )
        return

    @pytest.fixture(scope='class')
    def complex_model(self) -> Model[TorchTuple, TorchData]:
        """Fixture of a complex model wrapped with Model."""
        cpu = torch.device('cpu')
        return Model(MLP(), name='mlp_model', device=cpu, torch_compile=False)

    def test_call_autocasts_to_the_given_type(self) -> None:
        """Test the forward runs under autocast to the given type."""
        cpu = torch.device('cpu')
        model = Model(
            MLP(), name='bf16_model', device=cpu, mixed_precision=torch.bfloat16
        )

        model(TorchTuple(torch.randn(1, 1)))

        self.mock_autocast.assert_called_once_with(
            device_type='cpu', dtype=torch.bfloat16, enabled=True
        )

    def test_auto_device_without_available_accelerator(self, mocker) -> None:
        """Test the automatic device is the CPU without an accelerator."""
        mocker.patch(
            'torch.accelerator.current_accelerator',
            return_value=torch.device('cuda'),
        )
        mocker.patch('torch.accelerator.is_available', return_value=False)

        model = Model(MLP(), name='default_device_model')

        assert model.device == torch.device('cpu')

    def test_other_mixed_precision_types_raise(self) -> None:
        """Test only float16, bfloat16 or None are accepted."""
        with pytest.raises(ValueError, match='mixed_precision must be'):
            Model(
                MLP(), device=torch.device('cpu'), mixed_precision=torch.float64
            )

    def test_model_increment_epoch(self, mocker, complex_model: Model) -> None:
        """Test increment_epoch advances the epoch and emits an event."""
        mock_event = mocker.patch.object(log_events, 'IncrementEpochEvent')

        complex_model.increment_epoch()

        assert complex_model.epoch == 1
        mock_event.assert_called_once_with(model_name='mlp_model', epoch=1)

    def test_load_state_sets_epoch(self, mocker, complex_model: Model) -> None:
        """Test load_state sets the epoch the checkpoint reports."""
        mocker.patch.object(complex_model.checkpoint, 'load', return_value=7)

        complex_model.load_state()

        assert complex_model.epoch == 7

    def test_no_dist(self, complex_model: Model) -> None:
        """Test module is exec_module outside distributed settings."""
        assert complex_model.module is complex_model.exec_module


class TestSWAModel:
    """Tests for the SWAModel wrapper."""

    @pytest.fixture
    def swa_model(self, mocker) -> SWAModel[TorchTuple, TorchData]:
        """Fixture for AveragedModel."""
        mocker.patch('torch.compile', side_effect=lambda x: x)
        cpu = torch.device('cpu')
        model = SWAModel(MLP(), name='swa_model', start_epoch=2, device=cpu)
        model._epoch = 2
        return model

    def test_init(self, swa_model) -> None:
        """Averaged module should exist and be frozen."""
        assert swa_model.start_epoch == 2
        for param in swa_model.averaged_module.parameters():
            assert param.requires_grad is False

    def test_update_parameters(self, swa_model) -> None:
        """Calling update_parameters should modify averaged weights."""
        initial = [p.clone() for p in swa_model.averaged_module.parameters()]
        for p in swa_model.module.parameters():
            p.data.zero_()

        swa_model.post_epoch_update()

        updated = list(swa_model.averaged_module.parameters())

        assert any(
            not torch.allclose(p0, p1)
            for p0, p1 in zip(initial, updated, strict=False)
        )

    def test_update_parameters_no_start(self, swa_model) -> None:
        """No averaging should occur if start_epoch is not reached yet."""
        swa_model._epoch = 0
        initial = [p.clone() for p in swa_model.averaged_module.parameters()]
        for p in swa_model.module.parameters():
            p.data.zero_()

        swa_model.post_epoch_update()

        updated = list(swa_model.averaged_module.parameters())

        assert all(
            torch.allclose(p0, p1)
            for p0, p1 in zip(initial, updated, strict=False)
        )

    def test_call_uses_base_model_when_not_inference(
        self, swa_model, mocker
    ) -> None:
        """Forward should use the base model outside inference mode."""
        spy = mocker.spy(swa_model.module, 'forward')
        x = torch.randn(1, 1)

        swa_model(TorchTuple(x))

        assert spy.called

    def test_call_uses_averaged_model_in_inference(
        self, swa_model, mocker
    ) -> None:
        """Forward should use averaged model in inference mode."""
        spy = mocker.spy(swa_model.averaged_module, 'forward')
        x = torch.randn(1, 1)

        with torch.inference_mode():
            swa_model(TorchTuple(x))

        assert spy.called

    def test_call_uses_base_model_before_start_epoch(
        self, swa_model, mocker
    ) -> None:
        """Forward should use averaged model in inference mode."""
        swa_model._epoch = swa_model.start_epoch - 1
        spy = mocker.spy(swa_model.module, 'forward')
        x = torch.randn(1, 1)

        with torch.inference_mode():
            swa_model(TorchTuple(x))

        assert spy.called


class TestEMAModel:
    """Tests for the EMAModel wrapper."""

    @pytest.fixture(scope='class')
    def ema_model(self, class_mocker) -> EMAModel[TorchTuple, TorchData]:
        """Fixture for the EMA model."""
        class_mocker.patch('torch.compile', side_effect=lambda x: x)
        cpu = torch.device('cpu')
        return EMAModel(MLP(), name='ema_model', decay=0.9, device=cpu)

    def test_init(self, ema_model) -> None:
        """Test class initialization."""
        fn = ema_model._get_multi_avg_fn()

        assert callable(fn)
        assert ema_model.decay == 0.9

    def test_ema_update_parameters_changes_weights(self, ema_model) -> None:
        """EMA update should modify averaged weights."""
        initial = [p.clone() for p in ema_model.averaged_module.parameters()]
        for p in ema_model.module.parameters():
            p.data.zero_()

        ema_model.post_batch_update()

        updated = list(ema_model.averaged_module.parameters())
        assert any(
            not torch.allclose(p0, p1)
            for p0, p1 in zip(initial, updated, strict=False)
        )

    def test_call_uses_averaged_model_in_inference(
        self, ema_model, mocker
    ) -> None:
        """Forward should use averaged model in inference mode."""
        spy = mocker.spy(ema_model.averaged_module, 'forward')
        x = torch.randn(1, 1)

        with torch.inference_mode():
            ema_model(TorchTuple(x))

        assert spy.called

    def test_call_uses_mixed_precision_in_inference(self) -> None:
        """Averaged model should run under autocast in inference mode."""
        cpu = torch.device('cpu')
        model = EMAModel(
            MLP(), name='ema_mixed', device=cpu, mixed_precision=torch.bfloat16
        )
        x = torch.randn(1, 1)

        with torch.inference_mode():
            outputs = model(TorchTuple(x))

        assert outputs.output.dtype == torch.bfloat16

    def test_call_uses_compiled_averaged_module(self, mocker) -> None:
        """Inference should use the compiled averaged module when requested."""
        compiled_average = mocker.Mock(return_value=TorchData(torch.zeros(1)))

        def fake_compile(module: torch.nn.Module) -> torch.nn.Module:
            # only the averaged module is in eval mode at construction
            return module if module.training else compiled_average

        mocker.patch('torch.compile', side_effect=fake_compile)
        cpu = torch.device('cpu')
        model = EMAModel(
            MLP(), name='ema_compiled', device=cpu, torch_compile=True
        )
        inputs = TorchTuple(torch.randn(1, 1))

        with torch.inference_mode():
            model(inputs)

        compiled_average.assert_called_once_with(inputs)


def test_auto_device_with_accelerator(mocker) -> None:
    """Test the current accelerator is selected with its index."""
    mocker.patch(
        'torch.accelerator.current_accelerator',
        return_value=torch.device('cuda'),
    )
    mocker.patch('torch.accelerator.is_available', return_value=True)
    mocker.patch('torch.accelerator.current_device_index', return_value=1)

    device = _get_auto_device()

    assert device == torch.device('cuda', 1)

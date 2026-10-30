"""Tests for the outputs that runners store, with the real ``apply_ops``."""

import dataclasses
import warnings

import torch

from drytorch import Test


@dataclasses.dataclass
class _InnerOutput:
    """Dataclass nested in the output of a model."""

    value: torch.Tensor


@dataclasses.dataclass
class _OuterOutput:
    """Output of a model holding a dataclass and an optional field."""

    value: torch.Tensor
    inner: _InnerOutput
    optional: torch.Tensor | None = None


def test_store_keeps_nested_dataclass_outputs(
    linear_model, identity_loader, zero_metrics_calc, run
) -> None:
    """Test outputs holding a dataclass or None are stored, without warnings."""
    runner = Test(
        linear_model,
        loader=identity_loader,
        metric=zero_metrics_calc,
        name='Stored',
    )
    output = _OuterOutput(torch.ones(1), _InnerOutput(torch.ones(1)))

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        runner._store(output)

    assert len(runner.outputs_list) == 1
    stored = runner.outputs_list[0]
    assert torch.equal(stored.inner.value, output.inner.value)
    assert stored.optional is None

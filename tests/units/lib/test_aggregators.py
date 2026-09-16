"""Tests for the "aggregators" module."""

import torch

import pytest

from drytorch.lib.aggregators import Aggregator, TorchMeanAccumulator


class TestTorchMeanAccumulator:
    """Tests for TorchMeanAccumulator."""

    def test_from_value(self) -> None:
        """Test the accumulator stores the sum and the number of elements."""
        acc = TorchMeanAccumulator.from_value(torch.ones((2, 2)))

        assert torch.is_tensor(acc.total)
        assert acc.total.item() == 4.0
        assert acc.count == 4

    def test_merge(self) -> None:
        """Test merging adds totals and counts."""
        acc = TorchMeanAccumulator.from_value(torch.tensor([1.0, 2.0]))
        other = TorchMeanAccumulator.from_value(torch.tensor([3.0]))

        acc.merge(other)

        assert acc.total.item() == 6.0
        assert acc.count == 3

    def test_reduce(self) -> None:
        """Test reduce returns the mean."""
        acc = TorchMeanAccumulator.from_value(torch.tensor([1.0, 2.0, 6.0]))

        result = acc.reduce()

        assert result.item() == 3.0

    def test_sync_without_distributed(self) -> None:
        """Test sync leaves the state unchanged when not distributed."""
        acc = TorchMeanAccumulator.from_value(torch.tensor([1.0, 3.0]))

        acc.sync()

        assert acc.total.item() == 4.0
        assert acc.count == 2


class TestAggregator:
    """Tests for Aggregator."""

    @pytest.fixture
    def aggregator(self) -> Aggregator[TorchMeanAccumulator]:
        """Fixture creating an aggregator with two accumulators."""
        return Aggregator(
            metric1=TorchMeanAccumulator.from_value(torch.tensor(2.0)),
            metric2=TorchMeanAccumulator.from_value(torch.tensor(4.0)),
        )

    def test_init(self, aggregator: Aggregator[TorchMeanAccumulator]) -> None:
        """Test initialization stores the accumulators by name."""
        assert aggregator.keys() == ['metric1', 'metric2']
        assert aggregator.accumulators['metric1'].total == 2.0
        assert aggregator.accumulators['metric2'].total == 4.0

    def test_merge_stored_name(
        self, aggregator: Aggregator[TorchMeanAccumulator]
    ) -> None:
        """Test merging an accumulator with a stored name merges it."""
        new = TorchMeanAccumulator.from_value(torch.tensor(6.0))

        aggregator.merge({'metric1': new})

        assert aggregator.accumulators['metric1'].total == 8.0
        assert aggregator.accumulators['metric1'].count == 2

    def test_merge_copies_new_name(
        self, aggregator: Aggregator[TorchMeanAccumulator]
    ) -> None:
        """Test merging an accumulator with a new name stores a copy."""
        new = TorchMeanAccumulator.from_value(torch.tensor(6.0))

        aggregator.merge({'metric3': new})

        assert aggregator.accumulators['metric3'] is not new
        assert aggregator.accumulators['metric3'].total == 6.0

    def test_merge_other_accumulators(
        self, aggregator: Aggregator[TorchMeanAccumulator]
    ) -> None:
        """Test merging another aggregator's accumulators matches names."""
        other = Aggregator(
            metric2=TorchMeanAccumulator.from_value(torch.tensor(2.0)),
            metric3=TorchMeanAccumulator.from_value(torch.tensor(1.0)),
        )

        aggregator.merge(other.accumulators)

        assert aggregator.keys() == ['metric1', 'metric2', 'metric3']
        assert aggregator.accumulators['metric2'].total == 6.0
        assert aggregator.accumulators['metric2'].count == 2

    def test_bool(self, aggregator: Aggregator[TorchMeanAccumulator]) -> None:
        """Test an aggregator is true only when it stores accumulators."""
        empty = Aggregator[TorchMeanAccumulator]()

        assert aggregator
        assert not empty

    def test_clear(self, aggregator: Aggregator[TorchMeanAccumulator]) -> None:
        """Test clearing removes all accumulators."""
        aggregator.clear()

        assert not aggregator.accumulators

    def test_sync_without_distributed(
        self, aggregator: Aggregator[TorchMeanAccumulator]
    ) -> None:
        """Test sync leaves the state unchanged when not distributed."""
        aggregator.sync()

        assert aggregator.accumulators['metric1'].total == 2.0
        assert aggregator.accumulators['metric1'].count == 1

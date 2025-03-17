"""Test abstract dumper."""

import pytest

from dry_torch.trackers.abstract_dumper import AbstractDumper


class TestAbstractDumper:
    """Tests for the AbstractDumper."""

    @pytest.fixture(autouse=True)
    @pytest.mark.parametrize('par_dir', [None, 'test'])
    def setup(self, par_dir) -> None:
        """"""
        self.par_dir = par_dir
        self.tracker = AbstractDumper(par_dir=par_dir)

    @pytest.mark.parametrize('par_dir', [None, 'test'])
    def test_par_dir(self,
                     start_experiment_event,
                     stop_experiment_event,
                     par_dir) -> None:
        """"""
        with pytest.raises(RuntimeError):
            _ = self.tracker.par_dir
        self.tracker.notify(start_experiment_event)
        if par_dir is None:
            assert self.tracker.par_dir == start_experiment_event.exp_dir
        else:
            assert self.tracker.par_dir == par_dir
        self.tracker.notify(stop_experiment_event)
        with pytest.raises(RuntimeError):
            _ = self.tracker.par_dir

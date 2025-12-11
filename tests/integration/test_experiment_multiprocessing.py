"""Integration tests for experiment multiprocessing safety."""

import multiprocessing
import pathlib

import pytest

from drytorch.core.experiment import Experiment, RunStatus


def _create_run_worker(args):
    """Worker function to create a run (module-level for pickling)."""
    tmp_path, base_id, exp_name = args
    exp_p = Experiment(config={}, name=exp_name, par_dir=tmp_path)
    run = exp_p.create_run(run_id=base_id)
    return run.id


def _update_status_worker(args):
    """Worker function to update the run status (module-level for pickling)."""
    tmp_path, exp_name, run_id, status = args
    exp_p = Experiment(config={}, name=exp_name, par_dir=tmp_path)
    run = exp_p.create_run(run_id=run_id)
    run.status = status
    run._update_registry()
    return None


class TestExperimentMultiprocessing:
    """Test multiprocessing safety for Experiment and RunRegistry."""

    @pytest.fixture(autouse=True)
    def par_dir(self, tmp_path) -> pathlib.Path:
        """Set up a temporary directory for parallel runs."""
        return pathlib.Path(tmp_path)

    def test_multiprocessing_ids(self, par_dir) -> None:
        """Test parallel creation results in unique IDs due to PID suffix."""
        base_id = 'parallel_run'
        exp_name = 'parallel_exp'
        num_processes = 4
        args = [(par_dir, base_id, exp_name) for _ in range(num_processes)]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(_create_run_worker, args)

        assert len(set(results)) == num_processes
        for res in results:
            assert res.startswith(f'{base_id}_')

        exp = Experiment(config={}, name=exp_name, par_dir=par_dir)
        runs = exp._registry.load_all()
        registry_ids = {r.id for r in runs}
        assert registry_ids == set(results)

    def test_status_update_race(self, par_dir) -> None:
        """Test that concurrent status updates do not corrupt the file."""
        exp_name = 'status_exp'
        run_id = 'status_run'
        status_list: list[RunStatus] = [
            'running',
            'completed',
            'failed',
            'running',
        ]
        args = [(par_dir, exp_name, run_id, status) for status in status_list]
        with multiprocessing.Pool(processes=4) as pool:
            pool.map(_update_status_worker, args)

        exp = Experiment(config={}, name=exp_name, par_dir=par_dir)
        runs = exp._registry.load_all()

        assert len(runs) == len(status_list)
        for run in runs:
            assert run.status in status_list

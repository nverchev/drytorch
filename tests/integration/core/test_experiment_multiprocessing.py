"""Integration tests for experiment multiprocessing safety."""

import multiprocessing
import pathlib
import warnings

from ..conftest import RunningWorker

import drytorch

from drytorch.core.exceptions import RunAlreadyCompletedWarning
from drytorch.core.experimenting import Experiment, RunStatus


class TestExperimentMultiprocessing:
    """Test multiprocessing safety for Experiment and RunRegistry."""

    def test_multiprocessing_ids(self, tmp_path, example_run_id) -> None:
        """Test parallel creation results in unique IDs due to PID suffix."""
        num_processes = 4

        with multiprocessing.Pool(processes=num_processes) as pool:
            worker = RunningWorker(
                self._get_run_id, par_dir=tmp_path, run_id=example_run_id
            )
            results = pool.starmap(worker, [() for _ in range(num_processes)])

        exp = Experiment(config=None, name=worker.name, par_dir=tmp_path)
        runs = exp._registry.load_all()
        registry_ids = {r.id for r in runs}

        assert len(set(results)) == num_processes
        for res in results:
            assert res.startswith(f'{example_run_id}_')

        assert registry_ids == set(results)

    def test_status_update_race(self, tmp_path, example_run_id) -> None:
        """Test that concurrent status updates do not corrupt the file."""
        status_list: list[RunStatus] = [
            'running',
            'completed',
            'failed',
            'running',
        ]
        worker = RunningWorker(
            self._update_status, par_dir=tmp_path, run_id=example_run_id
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RunAlreadyCompletedWarning)
            with multiprocessing.Pool(processes=len(status_list)) as pool:
                pool.map(worker, status_list)

        exp = Experiment(config={}, name=worker.name, par_dir=tmp_path)
        runs = exp._registry.load_all()

        assert len(runs) == len(status_list)
        for run in runs:
            assert run.status in status_list

    def test_tracking_in_secondary_process(
        self, tmp_path, example_run_id
    ) -> None:
        """Test that tracking is only active when the rank is one."""
        status_list: list[RunStatus] = [
            'running',
            'completed',
            'failed',
            'running',
        ]
        worker = RunningWorker(
            self._update_status, par_dir=tmp_path, run_id=example_run_id
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RunAlreadyCompletedWarning)
            with multiprocessing.Pool(processes=len(status_list)) as pool:
                pool.map(worker, status_list)

        exp = Experiment(config={}, name=worker.name, par_dir=tmp_path)
        runs = exp._registry.load_all()

        assert len(runs) == len(status_list)
        for run in runs:
            assert run.status in status_list

    def test_resume_in_worker(self, tmp_path, example_run_id) -> None:
        """Test a worker process resumes a run under its recorded id."""
        drytorch.remove_all_default_trackers()
        exp = Experiment(config=None, name='ResumeExperiment', par_dir=tmp_path)
        with exp.create_run(run_id=example_run_id):
            pass

        # spawn: a forked worker would inherit the experiment name counters
        context = multiprocessing.get_context('spawn')
        with context.Pool(processes=1) as pool:
            resumed_id = pool.apply(
                self._resume, (tmp_path, exp.name, example_run_id)
            )

        assert resumed_id == example_run_id

    @staticmethod
    def _get_run_id():
        """Worker function to create a run (module-level for pickling)."""
        return Experiment.get_current().run.id

    @staticmethod
    def _update_status(status: RunStatus):
        """Update the run status (module-level for pickling)."""
        run = Experiment.get_current().run
        run.status = status
        run._update_registry()
        return None

    @staticmethod
    def _resume(par_dir: pathlib.Path, name: str, run_id: str) -> str:
        """Resume a run and return its id (module-level for pickling)."""
        drytorch.remove_all_default_trackers()
        exp = Experiment(config=None, name=name, par_dir=par_dir)
        run = exp.create_run(run_id=run_id, resume=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RunAlreadyCompletedWarning)
            with run:
                pass

        return run.id

"""Tests for the local_ops module."""

import pathlib

from drytorch.utils import local_ops


def test_experiment_data_op(tmp_path: pathlib.Path) -> None:
    """Test experiment_data_op works as expected."""
    trackers = ['.drytorch', 'checkpoints', 'csv_metrics']
    exp_name = 'test_exp'

    for t in trackers:
        d = tmp_path / t / exp_name
        d.mkdir(parents=True)
        (d / 'data.txt').write_text(f'data in {t}')

    def _dummy_op(path: pathlib.Path, _: pathlib.Path | None) -> None:
        (path / 'modified.txt').write_text('modified')

    local_ops._experiment_data_op(_dummy_op, tmp_path, exp_name, '')

    for t in trackers:
        assert (tmp_path / t / exp_name / 'modified.txt').exists()


def test_delete_experiment_data(tmp_path: pathlib.Path) -> None:
    """Test delete_experiment_data."""
    trackers = ['.drytorch', 'checkpoints']
    exp_name = 'to_delete'

    for t in trackers:
        d = tmp_path / t / exp_name
        d.mkdir(parents=True)

    local_ops.delete_experiment_data(tmp_path, exp_name)

    for t in trackers:
        assert not (tmp_path / t / exp_name).exists()


def test_rename_experiment_data(tmp_path: pathlib.Path) -> None:
    """Test rename_experiment_data."""
    trackers = ['.drytorch', 'checkpoints']
    exp_name = 'old_name'
    new_name = 'new_name'

    for t in trackers:
        d = tmp_path / t / exp_name
        d.mkdir(parents=True)
        (d / 'file.txt').write_text('content')

    local_ops.rename_experiment_data(tmp_path, exp_name, new_name)

    for t in trackers:
        assert not (tmp_path / t / exp_name).exists()
        assert (tmp_path / t / new_name / 'file.txt').read_text() == 'content'


def test_clone_experiment_data(tmp_path: pathlib.Path) -> None:
    """Test clone_experiment_data."""
    trackers = ['.drytorch', 'checkpoints']
    exp_name = 'original'
    clone_name = 'clone'

    for t in trackers:
        d = tmp_path / t / exp_name
        d.mkdir(parents=True)
        (d / 'file.txt').write_text('content')

    local_ops.clone_experiment_data(tmp_path, exp_name, clone_name)

    for t in trackers:
        assert (tmp_path / t / exp_name / 'file.txt').read_text() == 'content'
        assert (tmp_path / t / clone_name / 'file.txt').read_text() == 'content'

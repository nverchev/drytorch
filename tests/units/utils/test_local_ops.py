"""Tests for the ``local_ops`` module."""

import pathlib

import pytest

from drytorch.core import exceptions
from drytorch.utils.local_ops import (
    _get_experiment_folders,
    clone_experiment_data,
    delete_experiment_data,
    rename_experiment_data,
)


def test_get_experiment_folders(tmp_path: pathlib.Path) -> None:
    """Test get_experiment_folders finds the folders of an experiment."""
    trackers = ['.drytorch', 'checkpoints', 'csv_metrics']
    exp_name = 'test_exp'

    for t in trackers:
        d = tmp_path / t / exp_name
        d.mkdir(parents=True)
        (d / 'data.txt').write_text(f'data in {t}')

    (tmp_path / 'checkpoints' / 'other_exp').mkdir()
    (tmp_path / 'loose_file.txt').write_text('not a tracker folder')

    folders = _get_experiment_folders(tmp_path, exp_name)

    expected = {tmp_path / t / exp_name for t in trackers}

    assert set(folders) == expected


def test_invalid_experiment_names_are_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """Test names that are not a plain folder name are rejected."""
    exp_name = 'valid_name'
    (tmp_path / 'checkpoints' / exp_name).mkdir(parents=True)
    errors = []

    for invalid in ('', '.', '..', 'nested/name', 'nested\\name'):
        try:
            delete_experiment_data(tmp_path, invalid)
        except ValueError:
            errors.append(invalid)

    assert len(errors) == 5
    assert (tmp_path / 'checkpoints' / exp_name).is_dir()


def test_rename_stops_before_touching_data(tmp_path: pathlib.Path) -> None:
    """Test a rename onto an existing folder leaves the data untouched."""
    trackers = ['.drytorch', 'checkpoints']
    exp_name = 'old_name'
    new_name = 'new_name'

    for t in trackers:
        d = tmp_path / t / exp_name
        d.mkdir(parents=True)
        (d / 'file.txt').write_text('content')

    (tmp_path / 'checkpoints' / new_name).mkdir()

    with pytest.raises(exceptions.FolderAlreadyExistsError):
        rename_experiment_data(tmp_path, exp_name, new_name)

    for t in trackers:
        assert (tmp_path / t / exp_name / 'file.txt').read_text() == 'content'

    assert not (tmp_path / '.drytorch' / new_name).exists()


def test_delete_experiment_data(tmp_path: pathlib.Path) -> None:
    """Test delete_experiment_data."""
    trackers = ['.drytorch', 'checkpoints']
    exp_name = 'to_delete'

    for t in trackers:
        d = tmp_path / t / exp_name
        d.mkdir(parents=True)

    delete_experiment_data(tmp_path, exp_name)

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

    rename_experiment_data(tmp_path, exp_name, new_name)

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

    clone_experiment_data(tmp_path, exp_name, clone_name)

    for t in trackers:
        assert (tmp_path / t / exp_name / 'file.txt').read_text() == 'content'
        assert (tmp_path / t / clone_name / 'file.txt').read_text() == 'content'


@pytest.mark.parametrize(
    'transfer', [clone_experiment_data, rename_experiment_data]
)
def test_transfer_of_a_missing_experiment_raises(
    tmp_path: pathlib.Path, transfer
) -> None:
    """Test cloning or renaming an experiment without data raises."""
    (tmp_path / 'checkpoints' / 'other').mkdir(parents=True)

    with pytest.raises(exceptions.ExperimentNotFoundError, match='missing'):
        transfer(tmp_path, 'missing', 'new')

    assert not (tmp_path / 'checkpoints' / 'new').exists()


def test_transfer_from_a_missing_parent_directory_raises(
    tmp_path: pathlib.Path,
) -> None:
    """Test a parent directory that does not exist holds no experiment."""
    with pytest.raises(exceptions.ExperimentNotFoundError):
        clone_experiment_data(tmp_path / 'missing_dir', 'exp', 'clone')


def test_clone_onto_an_existing_experiment_names_the_folder(
    tmp_path: pathlib.Path,
) -> None:
    """Test the error for an existing target names its folder."""
    for name in ('original', 'clone'):
        (tmp_path / 'checkpoints' / name).mkdir(parents=True)

    with pytest.raises(
        exceptions.FolderAlreadyExistsError, match=r'already exists: .*clone'
    ):
        clone_experiment_data(tmp_path, 'original', 'clone')


def test_delete_a_missing_experiment_raises(tmp_path: pathlib.Path) -> None:
    """Test deleting an experiment without data raises."""
    (tmp_path / 'checkpoints' / 'other').mkdir(parents=True)

    with pytest.raises(exceptions.ExperimentNotFoundError, match='missing'):
        delete_experiment_data(tmp_path, 'missing')

    assert (tmp_path / 'checkpoints' / 'other').exists()

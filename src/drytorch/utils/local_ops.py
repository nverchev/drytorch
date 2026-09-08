"""Module for managing local experiment folders."""

import pathlib
import shutil

from collections.abc import Callable

from drytorch.core import exceptions


__all__ = [
    'clone_experiment_data',
    'delete_experiment_data',
    'rename_experiment_data',
]

_TransferOp = Callable[[pathlib.Path, pathlib.Path], object]


def clone_experiment_data(
    par_dir: pathlib.Path | str, exp_name: str, new_exp_name: str
) -> None:
    """Clone local experiment data folders.

    Args:
        par_dir: parent directory of the experiment folders.
        exp_name: experiment name to clone.
        new_exp_name: name for the clone.
    """
    _transfer_experiment_data(shutil.copytree, par_dir, exp_name, new_exp_name)
    return


def delete_experiment_data(par_dir: pathlib.Path | str, exp_name: str) -> None:
    """Remove all local folders containing experiment data.

    Args:
        par_dir: parent directory of the experiment folders.
        exp_name: name of the experiment.
    """
    for exp_folder in _get_experiment_folders(par_dir, exp_name):
        shutil.rmtree(exp_folder)

    return


def rename_experiment_data(
    par_dir: pathlib.Path | str, exp_name: str, new_exp_name: str
) -> None:
    """Rename local folders containing experiment data.

    Args:
        par_dir: parent directory of the experiment folders.
        exp_name: existing experiment name.
        new_exp_name: new experiment name.
    """
    _transfer_experiment_data(
        pathlib.Path.rename, par_dir, exp_name, new_exp_name
    )
    return


def _get_experiment_folders(
    par_dir: pathlib.Path | str, exp_name: str
) -> list[pathlib.Path]:
    """Collect the folders containing the data of an experiment.

    Args:
        par_dir: parent directory of the experiment folders.
        exp_name: name of the experiment.

    Returns:
        The experiment folder inside each tracker folder.

    Raises:
        ValueError: if the experiment name is not a plain folder name.
    """
    _validate_name(exp_name)
    par_dir = pathlib.Path(par_dir)
    return [
        folder / exp_name
        for folder in par_dir.iterdir()
        if folder.is_dir() and (folder / exp_name).is_dir()
    ]


def _transfer_experiment_data(
    op: _TransferOp,
    par_dir: pathlib.Path | str,
    exp_name: str,
    new_exp_name: str,
) -> None:
    """Apply an operation moving experiment data to a new name.

    Args:
        op: operation to apply to the experiment data folders.
        par_dir: parent directory of the experiment folders.
        exp_name: name of the experiment.
        new_exp_name: new experiment name.

    Raises:
        ValueError: if an experiment name is not a plain folder name.
        FolderAlreadyExistsError: if a target folder already exists.
    """
    _validate_name(new_exp_name)
    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for exp_folder in _get_experiment_folders(par_dir, exp_name):
        new_folder = exp_folder.parent / new_exp_name
        if new_folder.exists():
            raise exceptions.FolderAlreadyExistsError(new_folder)

        pairs.append((exp_folder, new_folder))

    for exp_folder, new_folder in pairs:
        op(exp_folder, new_folder)

    return


def _validate_name(name: str) -> None:
    if not name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError(f'Invalid experiment name: {name!r}')

    return

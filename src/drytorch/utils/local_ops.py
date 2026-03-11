"""Module for managing local experiment folders."""

import pathlib
import shutil

from collections.abc import Callable


__all__ = [
    'clone_experiment_data',
    'delete_experiment_data',
    'rename_experiment_data',
]

_LocalOp = Callable[[pathlib.Path, pathlib.Path], None]


def _experiment_data_op(
    op: _LocalOp,
    par_dir: pathlib.Path | str,
    exp_name: str,
    new_exp_name: str,
) -> None:
    """Apply an operation to experiment data folders.

    Args:
        op: operation to apply to the experiment data folders.
        par_dir: parent directory of the experiment folders.
        exp_name: name of the experiment.
        new_exp_name: new experiment name.
    """
    par_dir = pathlib.Path(par_dir)
    for folder in par_dir.iterdir():
        if folder.is_dir():
            exp_folder = folder / exp_name
            if exp_folder.is_dir():
                op(exp_folder, folder / new_exp_name)

    return


def delete_experiment_data(par_dir: pathlib.Path | str, exp_name: str) -> None:
    """Remove all local folders containing experiment data.

    Args:
        par_dir: parent directory of the experiment folders.
        exp_name: name of the experiment.
    """

    def _delete(path: pathlib.Path, _: pathlib.Path) -> None:
        shutil.rmtree(path)
        return

    _experiment_data_op(_delete, par_dir, exp_name, '')
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

    def _rename(path: pathlib.Path, new_path: pathlib.Path) -> None:
        path.rename(new_path)
        return

    _experiment_data_op(_rename, par_dir, exp_name, new_exp_name)
    return


def clone_experiment_data(
    par_dir: pathlib.Path | str, exp_name: str, new_exp_name: str
) -> None:
    """Clone local experiment data folders.

    Args:
        par_dir: parent directory of the experiment folders.
        exp_name: experiment name to clone.
        new_exp_name: name for the clone.
    """

    def _clone(path: pathlib.Path, new_path: pathlib.Path) -> None:
        shutil.copytree(path, new_path)
        return

    _experiment_data_op(_clone, par_dir, exp_name, new_exp_name)
    return

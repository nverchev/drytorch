import atexit
import pathlib
import shutil

from dry_torch import tracking
from dry_torch import exceptions


def link_to_hydra(exp: tracking.Experiment) -> None:
    try:
        import hydra
    except ImportError:
        raise exceptions.LibraryNotAvailableError('hydra')

    # noinspection PyUnresolvedReferences
    str_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_dir = pathlib.Path(str_dir)
    hydra_link = exp.dir / 'hydra'
    hydra_link.mkdir(exist_ok=True)
    while True:
        link_name = exp.__class__._default_link_name()
        link_dir = hydra_link / link_name
        if link_dir.exists():
            continue
        if not hydra_dir.exists() and hydra_dir.is_dir():
            raise exceptions.LibraryNotSupportedError('hydra')
        else:
            link_dir.unlink(missing_ok=True)  # may replace broken link
            link_dir.symlink_to(hydra_dir, target_is_directory=True)
            break

    def copy_hydra_dir() -> None:
        link_dir.unlink()
        shutil.copytree(hydra_dir, link_dir)

    atexit.register(copy_hydra_dir)
    return

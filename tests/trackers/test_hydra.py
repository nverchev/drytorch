import pathlib

import hydra
import omegaconf

from src.dry_torch import Experiment
from src.dry_torch.trackers import hydra_backend
from src.dry_torch.trackers import builtin_logger


@hydra.main(version_base=None, config_path='conf', config_name='defaults')
def main(conf: omegaconf.DictConfig) -> None:
    assert conf.test_conf.pi == 3.14159
    print(conf)
    exp_pardir = pathlib.Path(__file__).parent.parent / 'experiments'
    exp = Experiment[None]('test hydra', pardir=exp_pardir)
    exp.register_tracker(hydra_backend.HydraLink(exp_pardir))
    exp.start()


def test_hydra():
    main()


if __name__ == '__main__':

    main()

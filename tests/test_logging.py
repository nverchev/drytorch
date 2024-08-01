import logging
from dry_torch import log_settings

logger = logging.getLogger('dry_torch')


def test_training_formatter():
    logger.log(log_settings.INFO_LEVELS.training, 'test')


def test_propagate_to_main():
    log_settings.propagate_to_root_logger()
    logger.log(log_settings.INFO_LEVELS.training, 'test')
    log_settings.set_default_logging()
    logger.log(log_settings.INFO_LEVELS.training, 'test')

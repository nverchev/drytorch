import logging
from dry_torch import logging

logger = logging.getLogger('dry_torch')


def test_training_formatter():
    logger.log(default_logging.INFO_LEVELS.training, 'test')


def test_propagate_to_main():
    default_logging.propagate_to_main_logger()
    logger.log(default_logging.INFO_LEVELS.training, 'test')
    default_logging.set_default_logging()
    logger.log(default_logging.INFO_LEVELS.training, 'test')

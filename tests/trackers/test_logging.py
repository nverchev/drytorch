import logging

logger = logging.getLogger('dry_torch')


def test_training_formatter():
    logger.log(log_settings.INFO_LEVELS.training, 'test')


def test_propagate_to_main():
    log_settings.enable_propagation()
    logger.log(log_settings.INFO_LEVELS.training, 'test')
    log_settings.enable_default_handler()
    logger.log(log_settings.INFO_LEVELS.training, 'test')

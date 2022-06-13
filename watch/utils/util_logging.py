"""
TODO:
    - [ ] This needs to be cleaned up. Python logging is way too global for my
    taste. I would prefer something more instance level, but that will require
    some thought. The netharn.FitHarn has a way of accomplishing this that I
    reasonably like. That might be worth generalizing and porting here.  But
    for now, this will remain somewhat ad-hoc.
"""
import logging.config


def setup_logging(verbose=1):
    """
    Define logging level

    Args:
        verbose (int):
            Accepted values:
                * 0: no logging
                * 1: INFO level
                * 2: DEBUG level
    """

    log_med = "%(asctime)s-15s %(name)-32s [%(levelname)-8s] %(message)s"
    log_large = "%(asctime)s-15s %(name)-32s [%(levelname)-8s] "
    log_large += "(%(module)-17s) %(message)s"

    log_config = {}

    if verbose == 0:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "null": {"level": "DEBUG", "class": "logging.NullHandler"}
            },
            "loggers": {
                "watchlog": {
                    "handlers": ["null"],
                    "propagate": True,
                    "level": "INFO"
                }
            },
        }
    elif verbose == 1:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": log_med
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "loggers": {
                "watchlog": {
                    "handlers": ["console"],
                    "propagate": True,
                    "level": "INFO",
                }
            },
        }
    elif verbose == 2:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "verbose": {
                    "format": log_large
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "verbose",
                }
            },
            "loggers": {
                "watchlog": {
                    "handlers": ["console"],
                    "propagate": True,
                    "level": "DEBUG",
                }
            },
        }
    else:
        raise ValueError("'verbose' must be one of: 0, 1, 2")
    return log_config


def get_logger(verbose=1):
    logcfg = setup_logging(verbose)
    logging.config.dictConfig(logcfg)
    logger = logging.getLogger('watchlog')
    return logger

"""
TODO:
    - [ ] This needs to be cleaned up. Python logging is way too global for my
    taste. I would prefer something more instance level, but that will require
    some thought. The netharn.FitHarn has a way of accomplishing this that I
    reasonably like. That might be worth generalizing and porting here.  But
    for now, this will remain somewhat ad-hoc.
"""
import logging.config
import warnings
from logging import CRITICAL, ERROR, WARNING, INFO, DEBUG


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

    if verbose > 2:
        warnings.warn(f'geowatch util_logging only accepts a maximum verbosity of 2. Reconfiguring {verbose} to 2.')
        verbose = 2

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


class PrintLogger:
    """
    Simple print-based logger that duck-types the logging.Logger class and
    "simply works" without configuration.

    Example:
        >>> from geowatch.utils.util_logging import *  # NOQA
        >>> logger = PrintLogger(level=logging.INFO)
        >>> logger.info('hello')
        >>> logger.debug('world')
        hello
        >>> logger = PrintLogger(level=logging.DEBUG)
        >>> logger.info('hello')
        >>> logger.debug('world')
        hello
        world
    """

    def __init__(self, name='<print-logger>', level=None, verbose=1):
        self.name = name
        if level is None:
            if verbose == 0:
                level = CRITICAL
            elif verbose == 1:
                level = INFO
            elif verbose == 2:
                level = DEBUG
        self.level = level
        self.parent = None
        self.propagate = True
        self.handlers = []
        self.disabled = False
        self._cache = {}

    def setLevel(self, level):
        raise NotImplementedError

    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.
        """
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.
        """
        if self.isEnabledFor(INFO):
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.
        """
        if self.isEnabledFor(WARNING):
            self._log(WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.
        """
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Convenience method for logging an ERROR with exception information.
        """
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.
        """
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """
        Log 'msg % args' with the integer severity 'level'.
        """
        if not isinstance(level, int):
            raise TypeError("level must be an integer")
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)

    def findCaller(self, stack_info=False, stacklevel=1):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        raise NotImplementedError

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        """
        return msg

    def filter(self, record):
        return True

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False,
             stacklevel=1):
        """
        Low-level logging routine which creates a LogRecord and then calls
        all the handlers of this logger to handle the record.
        """
        import sys
        sinfo = None
        fn, lno, func = "(unknown file)", 0, "(unknown function)"
        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
        record = self.makeRecord(self.name, level, fn, lno, msg, args,
                                 exc_info, func, extra, sinfo)
        self.handle(record)

    def handle(self, record):
        """
        Call the handlers for the specified record.

        This method is used for unpickled records received from a socket, as
        well as those created locally. Logger-level filtering is applied.
        """
        if (not self.disabled) and self.filter(record):
            self.callHandlers(record)

    def addHandler(self, hdlr):
        """
        Add the specified handler to this logger.
        """
        raise NotImplementedError

    def removeHandler(self, hdlr):
        """
        Remove the specified handler from this logger.
        """
        raise NotImplementedError

    def hasHandlers(self):
        """
        See if this logger has any handlers configured.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. Return True if a handler was found, else False.
        Stop searching up the hierarchy whenever a logger with the "propagate"
        attribute set to zero is found - that will be the last logger which
        is checked for the existence of handlers.
        """
        raise NotImplementedError

    def callHandlers(self, record):
        print(record)
        # raise NotImplementedError

    def getEffectiveLevel(self):
        """
        Get the effective level for this logger.
        """
        return self.level

    def isEnabledFor(self, level):
        """
        Is this logger enabled for level 'level'?
        """
        if self.disabled:
            return False
        return level >= self.level

    def getChild(self, suffix):
        raise NotImplementedError

    def __repr__(self):
        level = logging.getLevelName(self.getEffectiveLevel())
        return '<%s %s (%s)>' % (self.__class__.__name__, self.name, level)

    # def __reduce__(self):
    #     raise NotImplementedError

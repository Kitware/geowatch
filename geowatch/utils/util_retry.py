"""
A reimplementation of the retry library [RetryLib]_ with some minor changes,
mainly logging default to a print statement rather than an actual logger.

References:
    .. [RetryLib] https://github.com/invl/retry
"""
import time
import random


class DummyLogger:
    def warning(self, msg, *args):
        print(msg % args)


class Retry:
    """
    Reimplementation of the retry internals
    """
    def __init__(self, exceptions=Exception, tries=-1, delay=0, backoff=1, max_delay=None, jitter=0, logger=None):
        if tries <= 0:
            raise NotImplementedError('tries <= 0 is not supported')
        self.tries = tries
        self.delay = delay
        self.exceptions = exceptions
        self.backoff = backoff
        if logger is None:
            logger = DummyLogger()
        self.logger = logger
        self.jitter = jitter
        if max_delay is None:
            max_delay = float('inf')
        self.max_delay = max_delay

    def __call__(self, func, *args, **kwargs):
        current_delay = self.delay
        for try_num in range(1, self.tries + 1):
            try:
                return func(*args, **kwargs)
            except self.exceptions as ex:
                if try_num >= self.tries:
                    self.logger.warning(f'Error on try {try_num}/{self.tries}. ex={ex!r}, giving up.')
                    raise
                else:
                    self.logger.warning(f'Error on try {try_num}/{self.tries}. ex={ex!r}, retry after delay={current_delay:0.2f} seconds')

            if isinstance(current_delay, tuple):
                current_delay += random.uniform(*self.jitter)
            else:
                current_delay += self.jitter

            time.sleep(current_delay)
            current_delay *= self.backoff
            current_delay = min(current_delay, self.max_delay)


def retry_call(f, fargs=None, fkwargs=None, exceptions=Exception, tries=-1,
               delay=0, max_delay=None, backoff=1, jitter=0, logger=None):
    """
    Retry API compatable

    CommandLine:
        xdoctest -m geowatch.utils.util_retry retry_call

    Example:
        >>> from geowatch.utils.util_retry import retry_call
        >>> _context = {'attempt': 0}
        >>> def f():
        >>>     _context['attempt'] += 1
        >>>     if _context['attempt'] <= 5:
        >>>         raise Exception
        >>>     return 1
        >>> import pytest
        >>> with pytest.raises(Exception):
        ...     result = retry_call(f, tries=2, delay=0.01)
        >>> result = retry_call(f, tries=4, delay=0.01)
    """
    retry_obj = Retry(
        tries=tries, delay=delay, backoff=backoff, exceptions=exceptions,
        jitter=jitter, max_delay=max_delay, logger=logger)
    fargs = fargs or []
    fkwargs = fkwargs or {}
    return retry_obj(f, *fargs, **fkwargs)

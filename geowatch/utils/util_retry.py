"""
A reimplementation of the retry library [RetryLib]_ with some minor changes,
mainly logging default to a print statement rather than an actual logger.

References:
    .. [RetryLib] https://github.com/invl/retry
"""
import time


class DummyLogger:
    def warning(self, msg, *args):
        print(msg % args)


class Retry:
    def __init__(self, tries, delay=0, backoff=1, exceptions=Exception, logger=None):
        self.tries = tries
        self.delay = delay
        self.exceptions = exceptions
        self.backoff = backoff
        if logger is None:
            logger = DummyLogger()
        self.logger = logger

    def __call__(self, func, *args, **kwargs):
        current_delay = self.delay
        for try_num in range(1, self.tries + 1):
            try:
                return func(*args, **kwargs)
            except self.exceptions as ex:
                self.logger.warning(f'ex={ex}: retry after delay={current_delay:0.2f} seconds')
                if try_num >= self.tries:
                    raise
            time.sleep(current_delay)
            current_delay *= self.backoff


def retry_call(f, fargs=None, fkwargs=None, exceptions=Exception, tries=-1,
               delay=0, max_delay=None, backoff=1, jitter=0, logger=None):
    """
    Retry API compatable

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
    if tries <= 0:
        raise NotImplementedError('tries <= 0 is not supported')
    if jitter != 0:
        raise NotImplementedError('jitter')
    if max_delay is not None:
        raise NotImplementedError('max_delay')

    retry_obj = Retry(tries, delay=delay, backoff=backoff, exceptions=exceptions)
    fargs = fargs or []
    fkwargs = fkwargs or {}
    return retry_obj(f, *fargs, **fkwargs)

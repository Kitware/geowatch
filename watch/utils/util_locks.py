import ubelt as ub
import weakref


class Superlock:
    """
    A thread and/or process lock

    The lockiest lock that ever did lock... or at least an attempt at it.

    This is experimental and not well tested.

    If lock_fpath is NoParam, uses a global shared process lock. If None, then
    no process lock is used.

    If thread_key is NoParam, uses a global shared thread lock. If None, then
    no thread lock is used.

    Otherwise locks with the same process_fpath OR thread_key will not execute
    concurrently, up to system limitations of the locking mechanisms.

    Example:
        >>> self = Superlock()
        >>> with self:
        >>>     print('non-concurent code')

    Example:
        >>> from watch.utils.util_locks import *  # NOQA
        >>> import ubelt as ub
        >>> lock1 = Superlock()
        >>> lock2 = Superlock()
        >>> assert lock1.acquire(timeout=10)
        >>> assert not lock2.acquire(timeout=0.01)
        >>> lock1.release()
        >>> assert lock2.acquire()
        >>> lock2.release()

    Example:
        >>> from watch.utils.util_locks import *  # NOQA
        >>> import ubelt as ub
        >>> lock1 = Superlock(thread_key=None)
        >>> lock2 = Superlock(thread_key=None)
        >>> assert lock1.acquire(timeout=10)
        >>> assert not lock2.acquire(timeout=0.01)
        >>> lock1.release()
        >>> assert lock2.acquire()
        >>> lock2.release()
    """

    THREAD_LOCKS = weakref.WeakValueDictionary()
    GLOBAL_THREAD_KEY = '__GLOBAL_THREAD_LOCK__'
    GLOBAL_APPNAME = 'fasteners_ext/file_locks'
    GLOBAL_LOCK_FNAME = 'superlock.lock'

    def __init__(self, lock_fpath=ub.NoParam, thread_key=ub.NoParam):
        import fasteners
        import threading
        if lock_fpath is ub.NoParam:
            lock_fpath = self.global_lock_fpath

        if thread_key is ub.NoParam:
            thread_key = self.GLOBAL_THREAD_KEY

        self.lock_fpath = lock_fpath
        self.thread_key = thread_key
        self.process_lock = None
        self.thread_lock = None

        if thread_key is not None:
            self.thread_lock = self.THREAD_LOCKS.get(thread_key, None)
            if self.thread_lock is None:
                self.thread_lock = threading.Lock()
                self.THREAD_LOCKS[thread_key] = self.thread_lock

        if lock_fpath is not None:
            self.process_lock = fasteners.InterProcessLock(lock_fpath)  #

    @property
    def global_lock_fpath(self):
        global_dpath = ub.Path.appdir(self.GLOBAL_APPNAME, type='cache').ensuredir()
        global_lock_fpath = global_dpath / self.GLOBAL_LOCK_FNAME
        return global_lock_fpath

    def acquire(self, blocking=True, timeout=None, delay=0.01, max_delay=0.1):
        got = []
        # FIXME: corner case when one aquires and the other doesn't
        if self.thread_lock is not None:
            thread_timeout = -1 if timeout is None else timeout
            gotten1 = self.thread_lock.acquire(blocking=blocking, timeout=thread_timeout)
            got.append(gotten1)
        if self.process_lock is not None:
            gotten2 = self.process_lock.acquire(
                blocking=blocking, timeout=timeout, delay=delay,
                max_delay=max_delay)
            got.append(gotten2)
        gotten = all(got)
        return gotten

    def release(self):
        if self.process_lock is not None:
            self.process_lock.release()
        if self.thread_lock is not None:
            self.thread_lock.release()

    def __enter__(self):
        gotten = self.acquire()
        assert gotten, 'should always be true'
        return self

    def __exit__(self, a, b, c):
        self.release()

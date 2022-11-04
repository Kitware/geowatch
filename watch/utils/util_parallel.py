import ubelt as ub
# from collections import deque


class BlockingJobQueue:
    """
    Helper to execute some number of processes in a separate thread.

    A call to submit will block if there is any available background workers
    until at least one of them finishes.

    The wait_until_finished should always be called at the end.

    Example:
        >>> from watch.utils.util_parallel import *  # NOQA
        >>> import time
        >>> import random
        >>> # Test with zero workers
        >>> N = 100
        >>> global_list = []
        >>> def background_job(i):
        >>>     time.sleep(random.random() * 0.001)
        >>>     global_list.append(f'Executed job {i:03d}')
        >>> self = BlockingJobQueue()
        >>> for i in range(100):
        >>>     self.submit(background_job, i)
        >>> self.wait_until_finished()
        >>> assert len(global_list) == N
        >>> assert sorted(global_list) == global_list
        >>> #
        >>> # xdoctest: +REQUIRES(env:TEST_BLOCKING_JOB_QUEUE_THREADS)
        >>> #
        >>> # Test the threaded case
        >>> global_list = []
        >>> def background_job(i):
        >>>     time.sleep(random.random() * 0.1 + 0.1)
        >>>     global_list.append(f'Executed job {i:03d}')
        >>> self = BlockingJobQueue(max_workers=10)
        >>> for i in range(100):
        >>>     if i == self.max_workers:
        >>>         assert len(self.jobs) == self.max_workers
        >>>     self.submit(background_job, i)
        >>> self.wait_until_finished()
        >>> assert len(global_list) == N
        >>> assert sorted(global_list) != global_list
    """

    def __init__(self, mode='thread', max_workers=0):
        self.max_workers = max_workers
        self.executor = ub.Executor(mode=mode, max_workers=max_workers)
        self.jobs = []

    def has_room(self):
        return len(self.jobs) >= max(1, self.max_workers)

    def _wait_for_room(self):
        # Wait until the pool has available workers
        while len(self.jobs) >= max(1, self.max_workers):
            new_active_jobs = []
            for job in self.jobs:
                if job.running():
                    new_active_jobs.append(job)
                else:
                    # Check that the result is ok
                    job.result()
            self.jobs = new_active_jobs

    def wait_until_finished(self):
        for job in self.jobs:
            job.result()

    def submit(self, func, *args, **kwargs):
        self._wait_for_room()
        job = self.executor.submit(func, *args, **kwargs)
        self.jobs.append(job)
        return job

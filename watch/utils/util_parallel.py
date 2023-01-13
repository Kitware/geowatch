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

    def wait_until_finished(self, desc=None):
        if desc is None:
            jobiter = self.jobs
        else:
            jobiter = ub.ProgIter(self.jobs, desc=desc)
        for job in jobiter:
            job.result()

    def submit(self, func, *args, **kwargs):
        self._wait_for_room()
        job = self.executor.submit(func, *args, **kwargs)
        self.jobs.append(job)
        return job


def coerce_num_workers(num_workers='auto', minimum=0):
    """
    Return some number of CPUs based on a chosen hueristic

    Args:
        num_workers (int | str):
            A special string code, or an exact number of cpus

        minimum (int): minimum workers we are allowed to return

    Returns:
        int : number of available cpus based on request parameters

    CommandLine:
        xdoctest -m watch.utils.util_parallel coerce_num_workers

    Example:
        >>> from watch.utils.util_parallel import *  # NOQA
        >>> print(coerce_num_workers('all'))
        >>> print(coerce_num_workers('avail'))
        >>> print(coerce_num_workers('auto'))
        >>> print(coerce_num_workers('all-2'))
        >>> print(coerce_num_workers('avail-2'))
        >>> print(coerce_num_workers('all/2'))
        >>> print(coerce_num_workers('min(all,2)'))
        >>> print(coerce_num_workers('[max(all,2)][0]'))
        >>> import pytest
        >>> with pytest.raises(Exception):
        >>>     print(coerce_num_workers('all + 1' + (' + 1' * 100)))
        >>> total_cpus = coerce_num_workers('all')
        >>> assert coerce_num_workers('all-2') == max(total_cpus - 2, 0)
        >>> assert coerce_num_workers('all-100') == max(total_cpus - 100, 0)
        >>> assert coerce_num_workers('avail') <= coerce_num_workers('all')
        >>> assert coerce_num_workers(3) == max(3, 0)
    """
    import numpy as np
    import psutil
    from watch.utils.util_eval import restricted_eval

    try:
        num_workers = int(num_workers)
    except Exception:
        pass

    if isinstance(num_workers, str):

        num_workers = num_workers.lower()

        if num_workers == 'auto':
            num_workers = 'avail-2'

        # input normalization
        num_workers = num_workers.replace('available', 'avail')

        local_dict = {}

        # prefix = 'avail'
        if 'avail' in num_workers:
            current_load = np.array(psutil.cpu_percent(percpu=True)) / 100
            local_dict['avail'] = np.sum(current_load < 0.5)
        local_dict['all_'] = psutil.cpu_count()

        if num_workers == 'none':
            num_workers = None
        else:
            expr = num_workers.replace('all', 'all_')
            # note: eval is not safe, using numexpr instead
            # limit chars even futher if eval is used
            if 1:
                # Mitigate attack surface by restricting builtin usage
                max_chars = 32
                builtins_passlist = ['min', 'max', 'round', 'sum']
                num_workers = restricted_eval(expr, max_chars, local_dict,
                                              builtins_passlist)
            else:
                import numexpr
                num_workers = numexpr.evaluate(expr, local_dict=local_dict,
                                               global_dict=local_dict)

    if num_workers is not None:
        num_workers = max(int(num_workers), minimum)

    return num_workers

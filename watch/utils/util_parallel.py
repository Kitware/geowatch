import ubelt as ub


class BlockingJobQueue(object):
    """
    Helper to execute some number of processes in a separate thread
    """
    def __init__(self, mode='thread', max_workers=0):
        self.max_workers = max_workers
        self.executor = ub.Executor(mode=mode, max_workers=max_workers)
        self.jobs = []

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

"""
Notes:
    It would be nice for this to support an rsync backend that could sync
    at the src/dst pair level. Not sure if this works.

References:
    https://unix.stackexchange.com/questions/133995/rsyncing-multiple-src-dest-pairs
    https://serverfault.com/questions/163859/using-rsync-as-a-queue
    https://unix.stackexchange.com/questions/602606/rsync-source-list-to-destination-list
"""
import ubelt as ub


class CopyManager:
    """
    Helper to execute multiple copy operations on a local filesystem.

    Args:

        workers (int): number of parallel workers to use

        mode (str): thread, process, or serial

        eager (bool):
            if True starts copying as soon as a job is submitted, otherwise it
            wait until run is called.

    Example:
        >>> import ubelt as ub
        >>> from watch.utils import copy_manager
        >>> dpath = ub.Path.appdir('watch', 'tests', 'copy_manager')
        >>> src_dpath = (dpath / 'src').ensuredir()
        >>> dst_dpath = (dpath / 'dst').delete()
        >>> src_fpaths = [src_dpath / 'file{}.txt'.format(i) for i in range(10)]
        >>> for fpath in src_fpaths:
        >>>     fpath.touch()
        >>> copyman = copy_manager.CopyManager(workers)
        >>> for fpath in src_fpaths:
        >>>     dst = fpath.augment(dpath=dst_dpath)
        >>>     copyman.submit(fpath, dst)
        >>> copyman.run()
        >>> assert len(dst_dpath.ls()) == len(src_dpath.ls())
    """

    def __init__(self, workers=0, mode='thread', eager=False):
        self._pool = ub.JobPool(mode=mode, max_workers=workers)
        self._unsubmitted = []
        self.eager = eager

    def __enter__(self):
        self._pool.__enter__()
        return self

    def __len__(self):
        return len(self._unsubmitted) + len(self._pool)

    def __exit__(self, a=None, b=None, c=None):
        return self._pool.__exit__(a, b, c)

    def submit(self, src, dst):
        task = {'src': src, 'dst': dst}
        if self.eager:
            self._pool.submit(_copy_worker, **task)
        else:
            self._unsubmitted.append(task)

    def run(self):
        from watch.utils import util_progress
        pman = util_progress.ProgressManager()
        with pman:
            for task in pman.progiter(self._unsubmitted, desc='submit copy jobs'):
                self._pool.submit(_copy_worker, **task)
            self._unsubmitted.clear()
            job_iter = self._pool.as_completed()
            prog = pman.progiter(job_iter, desc='collect copy jobs', total=len(self._pool))
            for job in prog:
                job.result()


def _copy_worker(src, dst):
    src = ub.Path(src)
    dst = ub.Path(dst)
    dst.parent.ensuredir()
    src.copy(dst)

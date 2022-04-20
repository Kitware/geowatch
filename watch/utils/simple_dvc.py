"""
A simplified Python DVC API
"""
import ubelt as ub
import os

"""
Python API to make DVC easier to work with
"""


class ChDir:
    """
    Context manager that changes the current working directory and then
    returns you to where you were.
    """
    def __init__(self, dpath):
        self.context_dpath = dpath
        self.orig_dpath = None

    def __enter__(self):
        self.orig_dpath = os.getcwd()
        os.chdir(self.context_dpath)
        return self

    def __exit__(self, a, b, c):
        os.chdir(self.orig_dpath)


class SimpleDVC():
    """
    A Simple DVC API

    Ignore:
        >>> # xdoctest: +REQUIRES(--dvc-test)
        >>> import sys, ubelt
        >>> from watch.utils.simple_dvc import *  # NOQA
        >>> dvc_dpath = SimpleDVC.demo_dpath(reset=0)
        >>> self = SimpleDVC(dvc_dpath)
        >>> a_file_fpath = dvc_dpath / 'a_file.txt'
        >>> if not a_file_fpath.exists():
        >>>     a_file_fpath.write_text('hello')
        >>> self.add(a_file_fpath)

    """

    def __init__(self, dvc_root=None):
        self.dvc_root = dvc_root

    @classmethod
    def demo_dpath(cls, reset=False):
        dvc_dpath = ub.Path.appdir('simple_dvc/test/test_dvc_repo')
        if reset:
            dvc_dpath.delete()
        if not dvc_dpath.exists():
            dvc_dpath.ensuredir()
            verbose = 2
            # Init empty git repo
            ub.cmd('git init --quiet', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('git config --local receive.denyCurrentBranch "warn"', cwd=dvc_dpath, verbose=verbose)
            # Init empty dvc repo
            ub.cmd('dvc init --quiet', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config core.autostage true', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config cache.type "symlink,hardlink,copy"', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config cache.shared group', cwd=dvc_dpath, verbose=verbose)
            ub.cmd('dvc config cache.protected true', cwd=dvc_dpath, verbose=verbose)
        return dvc_dpath

    @classmethod
    def coerce(cls, dvc_path):
        """
        Given a path inside DVC, finds the root.
        """
        dvc_root = cls.find_root(dvc_path)
        return cls(dvc_root)

    @classmethod
    def find_root(cls, path=None):
        if path is None:
            raise Exception('no way to find dvc root')
        # Need to find it from the path
        path = ub.Path(path)
        max_parts = len(path.parts)
        curr = path
        found = None
        for _ in range(max_parts + 1):
            cand = curr / '.dvc'
            if cand.exists():
                found = curr
                break
            curr = curr.parent
        return found

    def _ensure_root(self, paths):
        if self.dvc_root is None:
            self.dvc_root = self.find_root(paths[0])
            print('found new self.dvc_root = {!r}'.format(self.dvc_root))
        return self.dvc_root

    def add(self, path):
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        dvc_root = self._ensure_root(paths)
        rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in paths]
        with ChDir(dvc_root):
            dvc_command = ['add'] + rel_paths
            dvc_main.main(dvc_command)

        has_autostage = ub.cmd('dvc config core.autostage', cwd=dvc_root, check=True)['out'].strip() == 'true'
        if not has_autostage:
            raise NotImplementedError('Need autostage to complete the git commit')

    def git_commitpush(self, message='', pull_on_fail=True):
        """
        TODO: better name here?
        """
        dvc_root = self.dvc_root
        git_info3 = ub.cmd(f'git commit -m "{message}"', verbose=3, check=True, cwd=dvc_root)  # dangerous?
        assert git_info3['ret'] == 0
        try:
            git_info2 = ub.cmd('git push', verbose=3, check=True, cwd=dvc_root)
        except Exception:
            if pull_on_fail:
                git_info2 = ub.cmd('git pull', verbose=3, check=True, cwd=dvc_root)
                git_info2 = ub.cmd('git push', verbose=3, check=True, cwd=dvc_root)
                assert git_info2['ret'] == 0
            else:
                raise

    def push(self, path, remote=None, recursive=False, jobs=None):
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        dvc_root = self._ensure_root(paths)
        extra_args = []
        if remote:
            extra_args += ['-r', remote]
        if jobs is not None:
            extra_args += ['--jobs', str(jobs)]
        if recursive:
            extra_args += ['--recursive']

        with ChDir(dvc_root):
            dvc_command = ['push'] + extra_args + [str(p.relative_to(dvc_root)) for p in paths]
            dvc_main.main(dvc_command)

    def pull(self, path, remote=None, recursive=False, jobs=None):
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        dvc_root = self._ensure_root(paths)
        extra_args = []
        if remote:
            extra_args += ['-r', remote]
        if jobs is not None:
            extra_args += ['--jobs', str(jobs)]
        if recursive:
            extra_args += ['--recursive']

        with ChDir(dvc_root):
            dvc_command = ['pull'] + extra_args + [str(p.relative_to(dvc_root)) for p in paths]
            dvc_main.main(dvc_command)

    def unprotect(self, path):
        from dvc import main as dvc_main
        paths = list(map(ub.Path, _ensure_iterable(path)))
        dvc_root = self._ensure_root(paths)
        rel_paths = [os.fspath(p.relative_to(dvc_root)) for p in paths]
        with ChDir(dvc_root):
            dvc_command = ['unprotect'] + rel_paths
            dvc_main.main(dvc_command)

    def is_tracked(self, path):
        path = ub.Path(path)
        tracker_fpath = self.find_file_tracker(path)
        if tracker_fpath is not None:
            return True
        else:
            tracker_fpath = self.find_dir_tracker(path)
            if tracker_fpath is not None:
                raise NotImplementedError

    # @classmethod
    # def find_dvc_tracking_fpath(cls, path):

    def find_file_tracker(cls, path):
        assert not path.name.endswith('.dvc')
        tracker_fpath = path.augment(tail='.dvc')
        if tracker_fpath.exists():
            return tracker_fpath

    def find_dir_tracker(cls, path):
        # Find if an ancestor parent dpath is tracked
        prev = path
        dpath = path.parent
        while (not (dpath / '.dvc').exists()) and prev != dpath:
            tracker_fpath = dpath.augment(tail='.dvc')
            if tracker_fpath.exists():
                return tracker_fpath
            prev = dpath
            dpath = dpath.parent


def _ensure_iterable(inputs):
    return inputs if ub.iterable(inputs) else [inputs]

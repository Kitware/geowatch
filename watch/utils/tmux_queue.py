"""
A very simple queue based on tmux and bash
"""
import pathlib
import ubelt as ub
import itertools as it
import stat
import os


class TMUXQueue(ub.NiceRepr):
    """
    A lightweight, but limited job queue

    Example:
        >>> TMUXQueue('foo', 'foo')
    """
    def __init__(self, name, dpath, environ=None):
        self.name = name
        self.environ = environ
        self.dpath = pathlib.Path(dpath)
        self.fpath = self.dpath / (name + '.sh')
        self.header = ['#!/bin/bash']
        self.commands = []

    def __nice__(self):
        return f'{self.name} - {len(self.commands)}'

    def finalize_text(self):
        script = self.header
        if self.environ:
            script = script + [
                f'export {k}="{v}"' for k, v in self.environ.items()]
        script = script + self.commands
        text = '\n'.join(script)
        return text

    def submit(self, command):
        self.commands.append(command)

    def write(self):
        text = self.finalize_text()
        with open(self.fpath, 'w') as file:
            file.write(text)
        os.chmod(self.fpath, (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR |
            stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP))
        return self.fpath


class TMUXMultiQueue(ub.NiceRepr):
    """
    Create multiple sets of jobs to start in detatched tmux sessions

    Example:
        >>> from watch.utils.tmux_queue import *  # NOQA
        >>> self = TMUXMultiQueue('foo', 2)
        >>> print('self = {!r}'.format(self))
        >>> self.submit('echo hello')
        >>> self.submit('echo world')
        >>> self.submit('echo foo')
        >>> self.submit('echo bar')
        >>> self.submit('echo bazbiz')
        >>> self.write()
        >>> self.rprint()
    """
    def __init__(self, name, size=1, environ=None, dpath=None, gres=None):
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('watch/tmux_queue')
        self.dpath = pathlib.Path(dpath)
        self.name = name
        self.size = size
        self.environ = environ
        self.fpath = self.dpath / f'run_queues_{self.name}.sh'

        per_worker_environs = [environ] * size
        if gres:
            # TODO: more sophisticated GPU policy?
            per_worker_environs = [
                ub.dict_union(e, {
                    'CUDA_VISIBLE_DEVICES': str(cvd),
                })
                for cvd, e in zip(gres, per_worker_environs)]

        self.workers = [
            TMUXQueue(
                name='queue_{}_{}'.format(self.name, worker_idx),
                dpath=self.dpath,
                environ=e
            )
            for worker_idx, e in enumerate(per_worker_environs)
        ]
        self._worker_cycle = it.cycle(self.workers)

    def __nice__(self):
        return ub.repr2(self.workers)

    def __iter__(self):
        yield from self._worker_cycle

    def submit(self, command):
        return next(self._worker_cycle).submit(command)

    def finalize_text(self):
        # Create a driver script
        driver_lines = [ub.codeblock(
            '''
            #!/bin/bash
            # Driver script to start the tmux-queue
            echo "submitting jobs"
            ''')]
        for queue in self.workers:
            # run_command_in_tmux_queue(command, name)
            part = ub.codeblock(
                f'''
                ### Run Queue: {queue.name}
                tmux new-session -d -s {queue.name} "bash"
                tmux send -t {queue.name} "source {queue.fpath}" Enter
                ''').format()
            driver_lines.append(part)
        driver_lines += ['echo "jobs submitted"']
        driver_text = '\n\n'.join(driver_lines)
        return driver_text

    def write(self):
        text = self.finalize_text()
        for queue in self.workers:
            queue.write()
        with open(self.fpath, 'w') as file:
            file.write(text)
        os.chmod(self.fpath, (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR |
            stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP))
        return self.fpath

    def run(self):
        return ub.cmd(f'bash {self.fpath}', verbose=3, check=True)

    def rprint(self):
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.console import Console
        console = Console()
        for queue in self.workers:
            code = queue.finalize_text()
            console.print(Panel(Syntax(code, 'bash'), title=str(queue.fpath)))
        code = self.finalize_text()
        console.print(Panel(Syntax(code, 'bash'), title=str(self.fpath)))

"""
A very simple queue based on tmux and bash

It should be possible to add more functionality, such as:

    - [x] A linear job queue - via one tmux shell

    - [x] Mulitple linear job queues - via multiple tmux shells

    - [x] Ability to query status of jobs - tmux script writes status to a
          file, secondary thread reads is.

    - [ ] Central scheduler - given that we can know when a job is done
          a central scheduling process can run in the background, check
          the status of existing jobs, and spawn new jobs

    - [ ] Dependencies between jobs - given a central scheduler, it can
          only spawn a new job if a its dependencies have been met.

    - [ ] GPU resource requirements - if a job indicates how much of a
          particular resources it needs, the scheduler can only schedule the
          next job if it "fits" given the resources taken by the current
          running jobs.

    - [ ] Duck typed API that uses Slurm if available. Slurm is a robust
          full featured queuing system. If it is available we should
          make it easy for the user to swap the tmux queue for slurm.

    - [ ] Duck typed API that uses subprocesses. Tmux is not always available,
          we could go even lighter weight and simply execute a subprocess that
          does the same thing as the linear queue. The downside is you don't
          get the nice tmux way of looking at the status of what the jobs are
          doing, but that doesn't matter in debugged automated workflows, and
          this does seem like a nice simple utility. Doesnt seem to exist
          elsewhere either, but my search terms might be wrong.
"""
import ubelt as ub
import itertools as it
import stat
import os
import json


class TMUXLinearQueue(ub.NiceRepr):
    """
    A linear job queue

    Example:
        >>> TMUXLinearQueue('foo', 'foo')
    """
    def __init__(self, name, dpath, environ=None):
        self.name = name
        self.environ = environ
        self.dpath = ub.Path(dpath)
        self.fpath = self.dpath / (name + '.sh')
        self.header = '#!/bin/bash'
        self.state_fpath = self.dpath / 'job_state_{}.txt'.format(self.name)
        self.commands = []

    def __nice__(self):
        return f'{self.name} - {len(self.commands)}'

    def finalize_text(self):
        script = [self.header]

        def _mark_state(state):
            b = base_state.copy()
            b.update(state)
            text = json.dumps(b)
            script.append(r"printf '{}\n' > {}".format(text, self.state_fpath))

        base_state = {
            'status': 'init',
            'finished': 0,
            'total': len(self.commands),
            'name': self.name,
        }
        _mark_state({'status': 'init', 'finished': 0})
        if self.environ:
            _mark_state({'status': 'set_environ', 'finished': 0})
            script.extend([
                f'export {k}="{v}"' for k, v in self.environ.items()])

        for num, command in enumerate(self.commands):
            _mark_state({'status': 'run', 'finished': num})
            # TODO: Check if command failed and mark the state
            script.append(command)

        _mark_state({'status': 'done', 'finished': num + 1})
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
        >>> self.submit('echo hello && sleep 0.5')
        >>> self.submit('echo world && sleep 0.5')
        >>> self.submit('echo foo && sleep 0.5')
        >>> self.submit('echo bar && sleep 0.5')
        >>> self.submit('echo bazbiz && sleep 0.5')
        >>> self.write()
        >>> self.rprint()
        >>> if ub.find_exe('tmux'):
        >>>     self.run()
        >>>     self.monitor()
    """
    def __init__(self, name, size=1, environ=None, dpath=None, gres=None):
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('watch/tmux_queue')
        self.dpath = ub.Path(dpath)
        self.name = name
        self.size = size
        if environ is None:
            environ = {}
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
            TMUXLinearQueue(
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
        if not ub.find_exe('tmux'):
            raise Exception('tmux not found')
        return ub.cmd(f'bash {self.fpath}', verbose=3, check=True)

    def monitor(self):
        """
        Monitor progress until the jobs are done
        """
        import time
        from rich.live import Live
        from rich.table import Table

        def update_status_table():
            # https://rich.readthedocs.io/en/stable/live.html
            table = Table()
            table.add_column('name')
            table.add_column('status')
            table.add_column('finished')
            table.add_column('total')

            finished = True

            for worker in self.workers:
                color = ''
                try:
                    state = json.loads(worker.state_fpath.read_text())
                except Exception:
                    finished = False
                    state = {
                        'name': worker.name, 'status': 'unknown',
                        'total': len(worker.commands), 'finished': -1
                    }
                    color = '[red]'
                else:
                    finished &= (state['status'] == 'done')
                    if state['status'] == 'done':
                        color = '[green]'
                table.add_row(
                    state['name'],
                    state['status'],
                    f"{color}{state['finished']}",
                    f"{state['total']}",
                )
            return table, finished

        table, finished = update_status_table()
        with Live(table, refresh_per_second=4) as live:
            while not finished:
                time.sleep(0.4)
                table, finished = update_status_table()
                live.update(table)

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

"""
A very simple queue based on tmux and bash

It should be possible to add more functionality, such as:

    - [x] A linear job queue - via one tmux shell

    - [x] Mulitple linear job queues - via multiple tmux shells

    - [x] Ability to query status of jobs - tmux script writes status to a
          file, secondary thread reads is.

    - [ ] Unique identifier per queue

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
import uuid


class PathIdentifiable(ub.NiceRepr):
    """
    An object that has an name, unique-rootid (to indicate the parent object),
    and directory where it can write things to. Probably could clean this logic
    up.
    """
    def __init__(self, name='', rootid=None, dpath=None):
        if rootid is None:
            rootid = str(ub.timestamp()) + '_' + ub.hash_data(uuid.uuid4())[0:8]
        self.name = name
        self.rootid = rootid
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('tmux_queue', self.pathid)
        self.dpath = ub.Path(dpath)

    def __nice__(self):
        return '{}'.format(self.pathid)

    @property
    def pathid(self):
        """ A path-safe identifier for file names """
        return '{}_{}'.format(self.name, self.rootid)


class TMUXLinearQueue(PathIdentifiable):
    """
    A linear job queue

    Example:
        >>> TMUXLinearQueue('foo', 'foo')
    """
    def __init__(self, name='', dpath=None, rootid=None, environ=None):
        super().__init__(name=name, dpath=dpath, rootid=rootid)
        self.fpath = self.dpath / (self.pathid + '.sh')
        self.state_fpath = self.dpath / 'job_state_{}.txt'.format(self.pathid)
        self.environ = environ
        self.header = '#!/bin/bash'
        self.header_commands = []
        self.commands = []

    def __nice__(self):
        return f'{self.pathid} - {len(self.commands)}'

    def finalize_text(self, with_status=True):
        script = [self.header]

        total = len(self.commands)

        if with_status:
            script.append(ub.codeblock(
                f'''
                # Init state to keep track of job progress
                let "_QUEUE_NUM_ERRORED=0"
                let "_QUEUE_NUM_FINISHED=0"
                _QUEUE_TOTAL={total}
                _QUEUE_STATUS=""
                '''))

        def _mark_status(status):
            # be careful with json formatting here
            if with_status:
                script.append(ub.codeblock(
                    '''
                    _QUEUE_STATUS="{}"
                    ''').format(status))
                json_parts = [
                    '"{}": "{}"'.format('status', '\'$_QUEUE_STATUS\''),
                    '"{}": {}'.format('finished', '\'$_QUEUE_NUM_FINISHED\''),
                    '"{}": {}'.format('errored', '\'$_QUEUE_NUM_ERRORED\''),
                    '"{}": {}'.format('total', '\'$_QUEUE_TOTAL\''),
                    '"{}": "{}"'.format('name', self.name),
                    '"{}": "{}"'.format('rootid', self.rootid),
                ]
                dump_code = 'printf \'{' + ', '.join(json_parts) + '}\\n\' > ' + str(self.state_fpath)
                script.append(dump_code)
                script.append('cat ' + str(self.state_fpath))

        _mark_status('init')
        if self.environ:
            _mark_status('set_environ')
            script.extend([
                f'export {k}="{v}"' for k, v in self.environ.items()])

        for command in self.header_commands:
            script.append('set -x')
            script.append(command)
            script.append('set +x')

        for num, command in enumerate(self.commands):
            _mark_status('run')
            script.append(ub.codeblock(
                '''
                #
                # Command {} / {}
                ''').format(num + 1, total))
            script.append('set -x')
            script.append(command)
            # Check command status and update the bash state
            if with_status:
                script.append(ub.codeblock(
                    '''
                    if [[ "$?" == "0" ]]; then
                        let "_QUEUE_NUM_FINISHED=_QUEUE_NUM_FINISHED+1"
                    else
                        let "_QUEUE_NUM_ERRORED=_QUEUE_NUM_ERRORED+1"
                    fi
                    '''))
            script.append('set +x')

        _mark_status('done')
        text = '\n'.join(script)
        return text

    def add_header_command(self, command):
        self.header_commands.append(command)

    def submit(self, command):
        # TODO: we could accept additional args here that modify how we handle
        # the command in the bash script we build (i.e. if the script is
        # allowed to fail or not)
        self.commands.append(command)

    def write(self):
        text = self.finalize_text()
        with open(self.fpath, 'w') as file:
            file.write(text)
        os.chmod(self.fpath, (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR |
            stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP))
        return self.fpath


class TMUXMultiQueue(PathIdentifiable):
    """
    Create multiple sets of jobs to start in detatched tmux sessions

    CommandLine:
        xdoctest -m watch.utils.tmux_queue TMUXMultiQueue

    Example:
        >>> from watch.utils.tmux_queue import *  # NOQA
        >>> self = TMUXMultiQueue(2, 'foo')
        >>> print('self = {!r}'.format(self))
        >>> self.submit('echo hello && sleep 0.5')
        >>> self.submit('echo world && sleep 0.5')
        >>> self.submit('echo foo && sleep 0.5')
        >>> self.submit('echo bar && sleep 0.5')
        >>> self.submit('echo spam && sleep 0.5')
        >>> self.submit('echo spam && sleep 0.5')
        >>> self.submit('echo err && false')
        >>> self.submit('echo spam && sleep 0.5')
        >>> self.submit('echo eggs && sleep 0.5')
        >>> self.submit('echo bazbiz && sleep 0.5')
        >>> self.write()
        >>> self.rprint()
        >>> if ub.find_exe('tmux'):
        >>>     self.run()
        >>>     self.monitor()
    """
    def __init__(self, size=1, name=None, dpath=None, rootid=None, environ=None,
                 gres=None):
        super().__init__(name=name, rootid=rootid, dpath=dpath)

        if environ is None:
            environ = {}
        self.size = size
        self.environ = environ
        self.fpath = self.dpath / f'run_queues_{self.name}.sh'
        self.gres = gres

        self._init_workers()

    def _init_workers(self):
        per_worker_environs = [self.environ] * self.size
        if self.gres:
            # TODO: more sophisticated GPU policy?
            per_worker_environs = [
                ub.dict_union(e, {
                    'CUDA_VISIBLE_DEVICES': str(cvd),
                })
                for cvd, e in zip(self.gres, per_worker_environs)]

        self.workers = [
            TMUXLinearQueue(
                name='queue_{}_{}'.format(self.name, worker_idx),
                rootid=self.rootid,
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

    def submit(self, command, index=None):
        """
        Args:
            index (int): if True, forces this job into a particular queue
        """
        if index is None:
            worker = next(self._worker_cycle)
        else:
            worker = self.workers[index]
        return worker.submit(command)

    def add_header_command(self, command):
        """
        Adds a header command run at the start of each queue
        """
        for worker in self.workers:
            worker.add_header_command(command)

    @property
    def total_jobs(self):
        return sum(len(worker.commands) for worker in self.workers)

    def finalize_text(self):
        # Create a driver script
        driver_lines = [ub.codeblock(
            '''
            #!/bin/bash
            # Driver script to start the tmux-queue
            echo "submitting {} jobs"
            ''').format(self.total_jobs)]
        for queue in self.workers:
            # run_command_in_tmux_queue(command, name)
            part = ub.codeblock(
                f'''
                ### Run Queue: {queue.pathid} with {len(queue.commands)} jobs
                tmux new-session -d -s {queue.pathid} "bash"
                tmux send -t {queue.pathid} "source {queue.fpath}" Enter
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

    def run(self, block=False):
        if not ub.find_exe('tmux'):
            raise Exception('tmux not found')
        self.write()
        ub.cmd(f'bash {self.fpath}', verbose=3, check=True)
        if block:
            return self.monitor()

    def monitor(self, refresh_rate=0.4):
        """
        Monitor progress until the jobs are done
        """
        import time
        from rich.live import Live
        from rich.table import Table

        def update_status_table():
            # https://rich.readthedocs.io/en/stable/live.html
            table = Table()
            columns = ['name', 'status', 'finished', 'errors', 'total']
            for col in columns:
                table.add_column(col)

            finished = True
            agg_state = {
                'name': 'agg',
                'status': '',
                'errored': 0,
                'finished': 0,
                'total': 0
            }

            for worker in self.workers:
                fin_color = ''
                err_color = ''
                try:
                    state = json.loads(worker.state_fpath.read_text())
                except Exception:
                    finished = False
                    state = {
                        'name': worker.name,
                        'status': 'unknown',
                        'total': len(worker.commands),
                        'finished': None,
                        'errored': None,
                    }
                    fin_color = '[yellow]'
                else:
                    finished &= (state['status'] == 'done')
                    if state['status'] == 'done':
                        fin_color = '[green]'

                    if (state['errored'] > 0):
                        err_color = '[red]'

                    agg_state['total'] += state['total']
                    agg_state['finished'] += state['finished']
                    agg_state['errored'] += state['errored']

                table.add_row(
                    state['name'],
                    state['status'],
                    f"{fin_color}{state['finished']}",
                    f"{err_color}{state['errored']}",
                    f"{state['total']}",
                )

            if not finished:
                agg_state['status'] = 'run'
            else:
                agg_state['status'] = 'done'

            if len(self.workers) > 1:
                table.add_row(
                    agg_state['name'],
                    agg_state['status'],
                    f"{agg_state['finished']}",
                    f"{agg_state['errored']}",
                    f"{agg_state['total']}",
                )
            return table, finished, agg_state

        table, finished, agg_state = update_status_table()
        with Live(table, refresh_per_second=4) as live:
            while not finished:
                time.sleep(refresh_rate)
                table, finished, agg_state = update_status_table()
                live.update(table)
        return agg_state

    def rprint(self, with_status=False, with_rich=0):
        """
        Print info about the commands, optionally with rich
        """
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.console import Console
        console = Console()
        for queue in self.workers:
            code = queue.finalize_text(with_status=with_status)
            if with_rich:
                console.print(Panel(Syntax(code, 'bash'), title=str(queue.fpath)))
                # console.print(Syntax(code, 'bash'))
            else:
                print(ub.highlight_code(f'# --- {str(queue.fpath)}', 'bash'))
                print(ub.highlight_code(code, 'bash'))

        code = self.finalize_text()
        console.print(Panel(Syntax(code, 'bash'), title=str(self.fpath)))

    def kill(self):
        # Kills all the tmux panes
        for queue in self.workers:
            print('\n\nqueue = {!r}'.format(queue))
            # First print out the contents for debug
            ub.cmd(f'tmux capture-pane -p -t "{queue.pathid}:0.0"', verbose=2)
            # Then kill it
            ub.cmd(f'tmux kill-session -t {queue.pathid}', verbose=2)

    def _tmux_current_sessions(self):
        # Kills all the tmux panes
        info = ub.cmd('tmux list-sessions')
        sessions = []
        for line in info['out'].split('\n'):
            line = line.strip()
            if line:
                session_id, rest = line.split(':', 1)
                sessions.append({
                    'id': session_id,
                    'rest': rest
                })
        return sessions


if 0:
    __tmux_notes__ = """
    # Useful tmux commands

    tmux list-commands


    tmux new-session -d -s {queue.pathid} "bash"
    tmux send -t {queue.pathid} "source {queue.fpath}" Enter

    tmux new-session -d -s my_session_id "bash"

    tmux list-sessions
    tmux list-panes -a
    tmux list-windows -a

    # This can query the content of the current pane
    tmux capture-pane -p -t "my_session_id:0.0"

    tmux attach-session -t my_session_id

    tmux kill-session -t my_session_id

    tmux list-windows -t my_session_id

    tmux capture-pane -t my_session_id
    tmux capture-pane --help
    -t my_session_id




    """

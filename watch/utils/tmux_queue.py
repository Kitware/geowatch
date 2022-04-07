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

    - [X] Dependencies between jobs - given a central scheduler, it can
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

    - [ ] Handle the case where some jobs need the GPU and others do not
"""
import ubelt as ub
# import itertools as it
import stat
import os
import uuid

from watch.utils import cmd_queue
from watch.utils import serial_queue


class TMUXMultiQueue(cmd_queue.Queue):
    """
    Create multiple sets of jobs to start in detatched tmux sessions

    CommandLine:
        xdoctest -m watch.utils.tmux_queue TMUXMultiQueue

    Example:
        >>> from watch.utils.serial_queue import *  # NOQA
        >>> self = TMUXMultiQueue(1, 'test-serial-queue')
        >>> job1 = self.submit('echo hi 1 && false')
        >>> job2 = self.submit('echo hi 2 && true')
        >>> job3 = self.submit('echo hi 3 && true', depends=job1)
        >>> self.rprint()
        >>> if ub.find_exe('tmux'):
        >>>     self.run()
        >>>     self.monitor()
        >>>     self.current_output()
        >>>     self.kill()

    Example:
        >>> from watch.utils.tmux_queue import *  # NOQA
        >>> self = TMUXMultiQueue(2, 'foo')
        >>> print('self = {!r}'.format(self))
        >>> job1 = self.submit('echo hello && sleep 0.5')
        >>> job2 = self.submit('echo world && sleep 0.5', depends=[job1])
        >>> job3 = self.submit('echo foo && sleep 0.5')
        >>> job4 = self.submit('echo bar && sleep 0.5')
        >>> job5 = self.submit('echo spam && sleep 0.5', depends=[job1])
        >>> job6 = self.submit('echo spam && sleep 0.5')
        >>> job7 = self.submit('echo err && false')
        >>> job8 = self.submit('echo spam && sleep 0.5')
        >>> job9 = self.submit('echo eggs && sleep 0.5', depends=[job8])
        >>> job10 = self.submit('echo bazbiz && sleep 0.5', depends=[job9])
        >>> self.write()
        >>> self.rprint()
        >>> if ub.find_exe('tmux'):
        >>>     self.run()
        >>>     self.monitor()
        >>>     self.current_output()
        >>>     self.kill()

    Ignore:
        >>> from watch.utils.tmux_queue import *  # NOQA
        >>> self = TMUXMultiQueue(2, 'foo', gres=[0, 1])
        >>> job1 = self.submit('echo hello && sleep 0.5')
        >>> job2 = self.submit('echo hello && sleep 0.5')
        >>> self.rprint()

        >>> from watch.utils.tmux_queue import *  # NOQA
        >>> self = TMUXMultiQueue(2, 'foo')
        >>> job1 = self.submit('echo hello && sleep 0.5')
        >>> self.sync()
        >>> job2 = self.submit('echo hello && sleep 0.5')
        >>> self.sync()
        >>> job3 = self.submit('echo hello && sleep 0.5')
        >>> self.sync()
        >>> self.rprint()

        >>> from watch.utils.tmux_queue import *  # NOQA
        >>> self = TMUXMultiQueue(2, 'foo')
        >>> job1 = self.submit('echo hello && sleep 0.5')
        >>> job2 = self.submit('echo hello && sleep 0.5')
        >>> job3 = self.submit('echo hello && sleep 0.5')
        >>> self.rprint()

    """
    def __init__(self, size=1, name=None, dpath=None, rootid=None, environ=None,
                 gres=None):
        super().__init__()

        if rootid is None:
            rootid = str(ub.timestamp().split('T')[0]) + '_' + ub.hash_data(uuid.uuid4())[0:8]
        self.name = name
        self.rootid = rootid
        self.pathid = '{}_{}'.format(self.name, self.rootid)
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('cmd_queue', self.pathid)
        self.dpath = ub.Path(dpath)

        if environ is None:
            environ = {}
        self.size = size
        self.environ = environ
        self.fpath = self.dpath / f'run_queues_{self.name}.sh'
        self.gres = gres

        self.jobs = []
        self.header_commands = []

        self._new_workers()

    def _new_workers(self, start=0):
        per_worker_environs = [self.environ] * self.size
        if self.gres:
            # TODO: more sophisticated GPU policy?
            per_worker_environs = [
                ub.dict_union(e, {
                    'CUDA_VISIBLE_DEVICES': str(cvd),
                })
                for cvd, e in zip(self.gres, per_worker_environs)]

        workers = [
            serial_queue.SerialQueue(
                name='queue_{}_{}'.format(self.name, worker_idx),
                rootid=self.rootid,
                dpath=self.dpath,
                environ=e
            )
            for worker_idx, e in enumerate(per_worker_environs, start=start)
        ]
        return workers

    def __nice__(self):
        return ub.repr2(self.jobs)

    def _semaphore_wait_command(self, flag_fpaths, msg):
        r"""
        TODO: use flock?

        Ignore:

            #  In queue 1
            flock /var/lock/lock1.lock python -c 'while True: print(".", end="")'

            #  In queue 2
            flock /var/lock/lock2.lock python -c 'while True: print(".", end="")'

            #  In queue 3
            flock /var/lock/lock1.lock echo "first lock finished" && \
                flock /var/lock/lock2.lock echo "second lock finished" && \
                    python -c "print('this command depends on lock1 and lock2 procs completing')"


            flock /var/lock/lock2.lock echo "second lock finished"

            flock /var/lock/lock1.lock /var/lock/lock2.lock -c python -c 'while True: print("hi")'
        """
        # TODO: use inotifywait
        conditions = ['[ ! -f {} ]'.format(p) for p in flag_fpaths]
        condition = ' || '.join(conditions)
        # TODO: count number of files that exist
        command = ub.codeblock(
            f'''
            printf "{msg} "
            while {condition};
            do
               sleep 1;
            done
            printf "finished {msg} "
            ''')
        return command

    def _semaphore_signal_command(self, flag_fpath):
        return ub.codeblock(
            f'''
            # Signal this worker is complete
            mkdir -p {flag_fpath.parent} && touch {flag_fpath}
            '''
        )

    def order_jobs(self):
        """
        Example:
            >>> from watch.utils.tmux_queue import *  # NOQA
            >>> self = TMUXMultiQueue(5, 'foo')
            >>> job1a = self.submit('echo hello && sleep 0.5')
            >>> job1b = self.submit('echo hello && sleep 0.5')
            >>> job2a = self.submit('echo hello && sleep 0.5', depends=[job1a])
            >>> job2b = self.submit('echo hello && sleep 0.5', depends=[job1b])
            >>> job3 = self.submit('echo hello && sleep 0.5', depends=[job2a, job2b])
            >>> self.rprint()

            self.run(block=True)

        Example:
            >>> from watch.utils.tmux_queue import *  # NOQA
            >>> self = TMUXMultiQueue(5, 'foo')
            >>> job0 = self.submit('true')
            >>> job1 = self.submit('true')
            >>> job2 = self.submit('true', depends=[job0])
            >>> job3 = self.submit('true', depends=[job1])
            >>> #job2c = self.submit('true', depends=[job1a, job1b])
            >>> #self.sync()
            >>> job4 = self.submit('true', depends=[job2, job3, job1])
            >>> job5 = self.submit('true', depends=[job4])
            >>> job6 = self.submit('true', depends=[job4])
            >>> job7 = self.submit('true', depends=[job4])
            >>> job8 = self.submit('true', depends=[job5])
            >>> job9 = self.submit('true', depends=[job6])
            >>> job10 = self.submit('true', depends=[job6])
            >>> job11 = self.submit('true', depends=[job7])
            >>> job12 = self.submit('true', depends=[job10, job11])
            >>> job13 = self.submit('true', depends=[job4])
            >>> job14 = self.submit('true', depends=[job13])
            >>> job15 = self.submit('true', depends=[job4])
            >>> job16 = self.submit('true', depends=[job15, job13])
            >>> job17 = self.submit('true', depends=[job4])
            >>> job18 = self.submit('true', depends=[job17])
            >>> job19 = self.submit('true', depends=[job14, job16, job17])
            >>> self.rprint()
            >>> # self.run(block=True)
        """
        import networkx as nx
        graph = self._dependency_graph()

        # Get rid of implicit dependencies
        try:
            reduced_graph = nx.transitive_reduction(graph)
        except Exception as ex:
            print('ex = {!r}'.format(ex))
            print('graph = {!r}'.format(graph))
            print(len(graph.nodes))
            print('graph.nodes = {}'.format(ub.repr2(graph.nodes, nl=1)))
            print('graph.edges = {}'.format(ub.repr2(graph.edges, nl=1)))
            print(len(graph.edges))
            print(graph.is_directed())
            print(nx.is_forest(graph))
            print(nx.is_directed_acyclic_graph(graph))
            simple_cycles = list(nx.cycles.simple_cycles(graph))
            print('simple_cycles = {}'.format(ub.repr2(simple_cycles, nl=1)))
            import xdev
            xdev.embed()
            print(cmd_queue.graph_str(graph))
            raise

        in_cut_nodes = set()
        out_cut_nodes = set()
        cut_edges = []
        for n in reduced_graph.nodes:
            # TODO: need to also check that the paths to a source node are
            # not unique, otherwise we dont need to cut the node, but extra
            # cuts wont matter, just make it less effiicent
            in_d = reduced_graph.in_degree[n]
            out_d = reduced_graph.out_degree[n]
            if in_d > 1:
                cut_edges.extend(list(reduced_graph.in_edges(n)))
                in_cut_nodes.add(n)
            if out_d > 1:
                cut_edges.extend(list(reduced_graph.out_edges(n)))
                out_cut_nodes.add(n)

        list(nx.dfs_labeled_edges(reduced_graph))

        # cut_nodes = out_cut_nodes | in_cut_nodes

        cut_notes = in_cut_nodes.copy()
        cut_notes.update([v for u, v in cut_edges])

        cut_graph = reduced_graph.copy()
        cut_graph.remove_edges_from(cut_edges)

        # Get all the node groups disconnected by the cuts
        condensed = nx.condensation(reduced_graph, nx.weakly_connected_components(cut_graph))

        if 0:
            from graphid.util import util_graphviz
            import kwplot
            kwplot.autompl()
            util_graphviz.show_nx(graph, fnum=1)
            util_graphviz.show_nx(reduced_graph, fnum=3)
            util_graphviz.show_nx(condensed, fnum=2)

        # Rank each condensed group, which defines
        # what order it is allowed to be executed in
        rankings = ub.ddict(set)
        condensed_order = list(nx.topological_sort(condensed))
        for c_node in condensed_order:
            members = set(condensed.nodes[c_node]['members'])
            ancestors = set(ub.flatten([nx.ancestors(reduced_graph, m) for m in members]))
            cut_in_ancestors = ancestors & in_cut_nodes
            cut_out_ancestors = ancestors & out_cut_nodes
            cut_in_members = members & in_cut_nodes
            rank = len(cut_in_members) + len(cut_out_ancestors) + len(cut_in_ancestors)
            for m in members:
                rankings[rank].update(members)

        # cmd_queue.graph_str(condensed, write=print)

        # Each rank defines a group that must itself be ordered
        # Ranks will execute sequentially, members within the
        # rank *might* be run in parallel
        ranked_job_groups = []
        for rank, group in sorted(rankings.items()):
            subgraph = graph.subgraph(group)
            # Only things that can run in parapellel are disconnected components
            parallel_groups = []
            for wcc in list(nx.weakly_connected_components(subgraph)):
                sub_subgraph = subgraph.subgraph(wcc)
                wcc_order = list(nx.topological_sort(sub_subgraph))
                parallel_groups.append(wcc_order)
            # Ranked bins
            # Solve a bin packing problem to partition these into self.size groups
            from watch.utils.util_kwarray import balanced_number_partitioning
            group_weights = list(map(len, parallel_groups))
            groupxs = balanced_number_partitioning(group_weights, num_parts=self.size)
            rank_groups = [list(ub.take(parallel_groups, gxs)) for gxs in groupxs]
            rank_groups = [g for g in rank_groups if len(g)]

            # Reorder each group to better agree with submission order
            rank_jobs = []
            for group in rank_groups:
                priorities = []
                for nodes in group:
                    nodes_index = min(graph.nodes[n]['index'] for n in nodes)
                    priorities.append(nodes_index)
                final_queue_order = list(ub.flatten(ub.take(group, ub.argsort(priorities))))
                final_queue_jobs = [graph.nodes[n]['job'] for n in final_queue_order]
                rank_jobs.append(final_queue_jobs)
            ranked_job_groups.append(rank_jobs)

        queue_workers = []
        flag_dpath = (self.dpath / 'semaphores')
        prev_rank_flag_fpaths = None
        for rank, rank_jobs in enumerate(ranked_job_groups):
            # Hack, abuse init workers each time to construct workers
            workers = self._new_workers(start=len(queue_workers))
            rank_workers = []
            for worker, jobs in zip(workers, rank_jobs):
                # Add a dummy job to wait for dependencies of this linear
                # queue

                if prev_rank_flag_fpaths:
                    command = self._semaphore_wait_command(prev_rank_flag_fpaths, msg=f"wait for previous rank {rank - 1}")
                    # Note: this should not be a real job
                    worker.submit(command, bookkeeper=1)

                for job in jobs:
                    worker.submit(job.command)

                rank_workers.append(worker)

            queue_workers.extend(rank_workers)

            # Add a dummy job at the end of each worker to signal finished
            rank_flag_fpaths = []
            num_rank_workers = len(rank_workers)
            for worker_idx, worker in enumerate(rank_workers):
                rank_flag_fpath = flag_dpath / f'rank_flag_{rank}_{worker_idx}_{num_rank_workers}.done'
                command = self._semaphore_signal_command(rank_flag_fpath)
                # Note: this should not be a real job
                worker.submit(command, bookkeeper=1)
                rank_flag_fpaths.append(rank_flag_fpath)
            prev_rank_flag_fpaths = rank_flag_fpaths

        # Overwrite workers with our new dependency aware workers
        for worker in queue_workers:
            for header_command in self.header_commands:
                worker.add_header_command(header_command)
        self.workers = queue_workers

    def add_header_command(self, command):
        """
        Adds a header command run at the start of each queue
        """
        self.header_commands.append(command)

    def finalize_text(self):
        self.order_jobs()
        # Create a driver script
        driver_lines = [ub.codeblock(
            '''
            #!/bin/bash
            # Driver script to start the tmux-queue
            echo "submitting {} jobs"
            ''').format(self.num_real_jobs)]
        for queue in self.workers:
            # run_command_in_tmux_queue(command, name)
            part = ub.codeblock(
                f'''
                ### Run Queue: {queue.pathid} with {len(queue)} jobs
                tmux new-session -d -s {queue.pathid} "bash"
                tmux send -t {queue.pathid} \\
                    "source {queue.fpath}" \\
                    Enter
                ''').format()
            driver_lines.append(part)
        driver_lines += ['echo "jobs submitted"']
        driver_text = '\n\n'.join(driver_lines)
        return driver_text

    def write(self):
        self.order_jobs()
        for queue in self.workers:
            queue.write()
        text = self.finalize_text()
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
            agg_state = self.monitor()
            if not agg_state['errored']:
                self.kill()
            return agg_state

    def serial_run(self):
        """
        Hack to run everything without tmux. This really should be a different
        "queue" backend.

        See Serial Queue instead
        """
        self.order_jobs()
        queue_fpaths = []
        for queue in self.workers:
            fpath = queue.write()
            queue_fpaths.append(fpath)
        for fpath in queue_fpaths:
            ub.cmd(f'{fpath}', verbose=3, check=True)

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
                state = worker.read_state()
                if state['status'] == 'unknown':
                    finished = False
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

    def rprint(self, with_status=False, with_gaurds=False, with_rich=0):
        """
        Print info about the commands, optionally with rich
        """
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.console import Console
        self.order_jobs()
        console = Console()
        for queue in self.workers:
            queue.rprint(with_status=with_status, with_gaurds=with_gaurds,
                         with_rich=with_rich)
            # code = queue.finalize_text(with_status=with_status)
            # if with_rich:
            #     console.print(Panel(Syntax(code, 'bash'), title=str(queue.fpath)))
            #     # console.print(Syntax(code, 'bash'))
            # else:
            #     print(ub.highlight_code(f'# --- {str(queue.fpath)}', 'bash'))
            #     print(ub.highlight_code(code, 'bash'))

        code = self.finalize_text()
        console.print(Panel(Syntax(code, 'bash'), title=str(self.fpath)))

    def current_output(self):
        for queue in self.workers:
            print('\n\nqueue = {!r}'.format(queue))
            # First print out the contents for debug
            ub.cmd(f'tmux capture-pane -p -t "{queue.pathid}:0.0"', verbose=3)

    def kill(self):
        # Kills all the tmux panes
        for queue in self.workers:
            print('\n\nqueue = {!r}'.format(queue))
            # First print out the contents for debug
            ub.cmd(f'tmux capture-pane -p -t "{queue.pathid}:0.0"', verbose=3)
            # Then kill it
            ub.cmd(f'tmux kill-session -t {queue.pathid}', verbose=0)

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

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


class BashJob(ub.NiceRepr):
    """
    A job meant to run inside of a larger bash file. Analog of SlurmJob
    """
    def __init__(self, command, name=None, depends=None, gpus=None, cpus=None):
        if depends is not None and not ub.iterable(depends):
            depends = [depends]

        self.name = name
        self.command = command
        self.depends = depends

    def __nice__(self):
        return self.name


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


class LinearBashQueue(PathIdentifiable):
    """
    A linear job queue written to a single bash file

    Example:
        >>> self = LinearBashQueue('foo', 'foo')
        >>> self.rprint()
    """
    def __init__(self, name='', dpath=None, rootid=None, environ=None, cwd=None):
        super().__init__(name=name, dpath=dpath, rootid=rootid)
        self.fpath = self.dpath / (self.pathid + '.sh')
        self.state_fpath = self.dpath / 'job_state_{}.txt'.format(self.pathid)
        self.environ = environ
        self.header = '#!/bin/bash'
        self.header_commands = []
        self.commands = []
        self.cwd = cwd

    def __nice__(self):
        return f'{self.pathid} - {len(self.commands)}'

    def finalize_text(self, with_status=True, with_gaurds=True):
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
            if with_gaurds:
                script.append('set -x')
            script.extend([
                f'export {k}="{v}"' for k, v in self.environ.items()])
            if with_gaurds:
                script.append('set +x')

        if self.cwd:
            script.append(f'cd {self.cwd}')

        for command in self.header_commands:
            if with_gaurds:
                script.append('set -x')
            script.append(command)
            if with_gaurds:
                script.append('set +x')

        for num, command in enumerate(self.commands):
            _mark_status('run')
            script.append(ub.codeblock(
                '''
                #
                # Command {} / {}
                ''').format(num + 1, total))
            if with_gaurds:
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
            if with_gaurds:
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

    def rprint(self, with_status=False, with_gaurds=False, with_rich=0):
        """
        Print info about the commands, optionally with rich
        """
        code = self.finalize_text(with_status=with_status,
                                  with_gaurds=with_gaurds)
        if with_rich:
            from rich.panel import Panel
            from rich.syntax import Syntax
            from rich.console import Console
            console = Console()
            console.print(Panel(Syntax(code, 'bash'), title=str(self.fpath)))
            # console.print(Syntax(code, 'bash'))
        else:
            print(ub.highlight_code(f'# --- {str(self.fpath)}', 'bash'))
            print(ub.highlight_code(code, 'bash'))


class TMUXMultiQueue(PathIdentifiable):
    """
    Create multiple sets of jobs to start in detatched tmux sessions

    CommandLine:
        xdoctest -m watch.utils.tmux_queue TMUXMultiQueue

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
        >>>     self.kill()

    Ignore:
        >>> from watch.utils.tmux_queue import *  # NOQA
        >>> self = TMUXMultiQueue(2, 'foo', gres=[0, 1])
        >>> job1 = self.submit('echo hello && sleep 0.5')
        >>> job2 = self.submit('echo hello && sleep 0.5')
        >>> self.rprint()

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

        self.jobs = []

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
            LinearBashQueue(
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

    def submit(self, command, **kwargs):
        """
        Args:
            index (int): if True, forces this job into a particular queue
        """
        name = kwargs.get('name', None)
        if name is None:
            name = kwargs['name'] = self.name + '-job-{}'.format(len(self.jobs))
        job = BashJob(command, **kwargs)
        self.jobs.append(job)
        return job
        # if index is None:
        #     worker = next(self._worker_cycle)
        # else:
        #     worker = self.workers[index]
        # return worker.submit(command)

    def _dependency_graph(self):
        """
        Example:
            >>> from watch.utils.tmux_queue import *  # NOQA
            >>> self = TMUXMultiQueue(5, 'foo')
            >>> job1a = self.submit('echo hello && sleep 0.5')
            >>> job1b = self.submit('echo hello && sleep 0.5')
            >>> job2a = self.submit('echo hello && sleep 0.5', depends=[job1a])
            >>> job2b = self.submit('echo hello && sleep 0.5', depends=[job1b])
            >>> job3 = self.submit('echo hello && sleep 0.5', depends=[job2a, job2b])
            >>> jobX = self.submit('echo hello && sleep 0.5', depends=[])
            >>> jobY = self.submit('echo hello && sleep 0.5', depends=[jobX])
            >>> jobZ = self.submit('echo hello && sleep 0.5', depends=[jobY])
            >>> graph = self._dependency_graph()

            print(graph_str(graph))
            #for wcc in list(nx.weakly_connected_components(graph)):
        """
        import networkx as nx
        graph = nx.DiGraph()
        for index, job in enumerate(self.jobs):
            graph.add_node(job.name, job=job, index=index)
        for index, job in enumerate(self.jobs):
            if job.depends:
                for dep in job.depends:
                    graph.add_edge(dep.name, job.name)
        return graph

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
        """
        import networkx as nx
        graph = self._dependency_graph()

        if 0:
            # TODO: for diamond shaped diagrams see if we can place locks on
            # certain processes in certain queues from continuing while they
            # wait for dependencies but also let other jobs run in parallel
            sources = [n for n, d in graph.in_degree if d == 0]
            sinks = [n for n, d in graph.out_degree if d == 0]

            st_paths = {}
            for s in sources:
                for t in sinks:
                    path = list(nx.all_simple_paths(graph, s, t))
                    st_paths[(s, t)] = path

        # Determine what jobs need to be grouped together
        wcc_groups = []
        for wcc in list(nx.weakly_connected_components(graph)):
            sub = graph.subgraph(wcc)
            if 0:
                print(nx.forest_str(nx.minimum_spanning_arborescence(sub)))
            wcc_order = list(nx.topological_sort(sub))
            wcc_groups.append(wcc_order)

        # Solve a bin packing problem to partition these into self.size groups
        from watch.utils.util_kwarray import balanced_number_partitioning
        group_weights = list(map(len, wcc_groups))
        groupxs = balanced_number_partitioning(group_weights, num_parts=self.size)
        node_groups = [list(ub.take(wcc_groups, gxs)) for gxs in groupxs]

        # Reorder each group to better agree with submission order
        final_orders = []
        for group in node_groups:
            priorities = []
            for nodes in group:
                nodes_index = min(graph.nodes[n]['index'] for n in nodes)
                priorities.append(nodes_index)

            final_queue_order = list(ub.flatten(ub.take(group, ub.argsort(priorities))))
            final_queue_jobs = [graph.nodes[n]['job'] for n in final_queue_order]
            final_orders.append(final_queue_jobs)

        # Submit each job to the linear queue in the correct order
        for worker, jobs in zip(self.workers, final_orders):
            worker.commands.clear()
            for job in jobs:
                worker.submit(job.command)

    def add_header_command(self, command):
        """
        Adds a header command run at the start of each queue
        """
        for worker in self.workers:
            worker.add_header_command(command)

    @property
    def total_jobs(self):
        return len(self.jobs)
        # sum(len(worker.commands) for worker in self.workers)

    def finalize_text(self):
        self.order_jobs()
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
            return self.monitor()

    def serial_run(self):
        """
        Hack to run everything without tmux. This really should be a different
        "queue" backend.
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
        self.order_jobs()
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


class SerialQueue:
    """
    Serial drop-in replacement for the tmux queue.

    TODO:
        - [ ] Move to a different file
        - [ ] Parallel non-tmux version
    """
    def __init__(self):
        self.commands = []

    def submit(self, command):
        self.commands.append(command)

    def finalize_text(self):
        text = '\n\n'.join(self.commands)
        return text

    def rprint(self):
        text = self.finalize_text()
        print(ub.highlight_code(text, 'bash'))


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


def graph_str(graph, with_labels=True, sources=None, write=None, ascii_only=False):
    """
    Attempt extension of forest_str

    Example:
        >>> from watch.utils.tmux_queue import *  # NOQA

        >>> import networkx as nx
        >>> graph = nx.DiGraph()
        >>> graph.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
        >>> graph.add_edges_from([
        >>>     ('a', 'd'),
        >>>     ('b', 'd'),
        >>>     ('c', 'd'),
        >>>     ('d', 'e'),
        >>>     ('f1', 'g'),
        >>>     ('f2', 'g'),
        >>> ])
        >>> graph_str(graph, write=print)
        >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        >>> print('\nForest Str')
        >>> print(nx.forest_str(graph))
        >>> print('\nGraph Str (should be identical)')
        >>> print(graph_str(graph))
        >>> print('modified')
        >>> graph.add_edges_from([
        >>>     (7, 1)
        >>> ])
        >>> graph_str(graph, write=print)

        >>> print('\n\n next')
        >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        >>> graph.add_edges_from([
        >>>     (7, 1),
        >>>     (2, 4),
        >>> ])
        >>> graph_str(graph, write=print)

        >>> print('\n\n next')
        >>> graph = nx.erdos_renyi_graph(5, 1.0)
        >>> graph_str(graph, write=print)
    """
    import networkx as nx

    printbuf = []
    if write is None:
        _write = printbuf.append
    else:
        _write = write

    # Define glphys
    # Notes on available box and arrow characters
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # https://stackoverflow.com/questions/2701192/triangle-arrow
    if ascii_only:
        glyph_empty = "+"
        glyph_newtree_last = "+-- "
        glyph_newtree_mid = "+-- "
        glyph_endof_forest = "    "
        glyph_within_forest = ":   "
        glyph_within_tree = "|   "

        glyph_directed_last = "L-> "
        glyph_directed_mid = "|-> "

        glyph_undirected_last = "L-- "
        glyph_undirected_mid = "|-- "
    else:
        glyph_empty = "╙"
        glyph_newtree_last = "╙── "
        glyph_newtree_mid = "╟── "
        glyph_endof_forest = "    "
        glyph_within_forest = "╎   "
        glyph_within_tree = "│   "

        glyph_directed_last = "└─╼ "
        glyph_directed_mid = "├─╼ "
        glyph_directed_backedge = '╾'

        glyph_undirected_last = "└── "
        glyph_undirected_mid = "├── "
        glyph_undirected_backedge = '─'

    print('graph = {!r}'.format(graph))
    if len(graph.nodes) == 0:
        _write(glyph_empty)
    else:
        is_directed = graph.is_directed()
        print('is_directed = {!r}'.format(is_directed))

        if is_directed:
            glyph_last = glyph_directed_last
            glyph_mid = glyph_directed_mid
            glyph_backedge = glyph_directed_backedge
            succ = graph.succ
            pred = graph.pred
        else:
            glyph_last = glyph_undirected_last
            glyph_mid = glyph_undirected_mid
            glyph_backedge = glyph_undirected_backedge
            succ = graph.adj
            pred = graph.adj

        if sources is None:
            if is_directed:
                # use real source nodes for directed trees
                sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
            else:
                # use arbitrary sources for undirected trees
                sources = []

            if len(sources) == 0:
                # no clear source, choose something
                sources = [
                    min(cc, key=lambda n: graph.degree[n])
                    for cc in nx.connected_components(graph)
                ]

        print('sources = {!r}'.format(sources))
        # Populate the stack with each source node, empty indentation, and mark
        # the final node. Reverse the stack so sources are popped in the
        # correct order.
        last_idx = len(sources) - 1
        stack = [(None, node, "", (idx == last_idx))
                 for idx, node in enumerate(sources)][::-1]

        seen_nodes = set()
        seen_edges = set()
        implicit_seen = set()
        while stack:
            parent, node, indent, this_islast = stack.pop()
            edge = (parent, node)

            if node is not Ellipsis:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                seen_edges.add(edge)

            if not indent:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if this_islast:
                    this_prefix = indent + glyph_newtree_last
                    next_prefix = indent + glyph_endof_forest
                else:
                    this_prefix = indent + glyph_newtree_mid
                    next_prefix = indent + glyph_within_forest

            else:
                # For individual tree edges distinguish between directed and
                # undirected cases
                if this_islast:
                    this_prefix = indent + glyph_last
                    next_prefix = indent + glyph_endof_forest
                else:
                    this_prefix = indent + glyph_mid
                    next_prefix = indent + glyph_within_tree

            if node is Ellipsis:
                label = ' ...'
                children = []
            else:
                if with_labels:
                    label = graph.nodes[node].get("label", node)
                else:
                    label = node
                children = [child for child in succ[node] if child not in seen_nodes]
                neighbors = set(pred[node])
                others = (neighbors - set(children)) - {parent}
                implicit_seen.update(others)
                in_edges = [(node, other) for other in others]
                if in_edges:
                    in_nodes = ', '.join([str(uv[1]) for uv in in_edges])
                    suffix = ' '.join(['', glyph_backedge, in_nodes])
                else:
                    suffix = ''

            _write(this_prefix + str(label) + suffix)

            # Push children on the stack in reverse order so they are popped in
            # the original order.
            idx = 1
            for idx, child in enumerate(children[::-1], start=1):
                next_islast = idx <= 1
                try_frame = (node, child, next_prefix, next_islast)
                stack.append(try_frame)

            if node in implicit_seen:
                # if have used this node in any previous implicit edges, then
                # write an outgoing "implicit" connection.
                next_islast = idx <= 1
                try_frame = (node, Ellipsis, next_prefix, next_islast)
                stack.append(try_frame)

    if write is None:
        # Only return a string if the custom write function was not specified
        return "\n".join(printbuf)

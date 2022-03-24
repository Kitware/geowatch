"""
Work in progress. The idea is to provide a TMUX queue and a SLURM queue that
provide a common high level API, even though functionality might diverge, the
core functionality of running processes asynchronously should be provided.

Notes:
    # Installing and configuring SLURM
    See git@github.com:Erotemic/local.git init/setup_slurm.sh
    Or ~/local/init/setup_slurm.sh in my local checkout

    SUBMIT COMMANDS WILL USE /bin/sh by default, not sure how to fix that
    properly. There are workarounds though.


CommandLine:
   xdoctest -m watch.utils.slurm_queue __doc__

Example:
    >>> from watch.utils.slurm_queue import *  # NOQA
    >>> dpath = ub.Path.appdir('slurm_queue/tests')
    >>> queue = SlurmQueue()
    >>> job0 = queue.submit(f'echo "here we go"', name='root job')
    >>> job1 = queue.submit(f'mkdir {dpath}', depends=[job0])
    >>> job2 = queue.submit(f'echo "result=42" > {dpath}/test.txt ', depends=[job1])
    >>> job3 = queue.submit(f'cat {dpath}/test.txt', depends=[job2])
    >>> queue.rprint()
    >>> # xdoctest: +REQUIRES(--run)
    >>> queue.run()
"""
import ubelt as ub

from watch.utils import cmd_queue  # NOQA


def _coerce_mem(mem):
    """
    Args:
        mem (int | str): integer number of megabytes or a parseable string

    Example:
        >>> from watch.utils.slurm_queue import *  # NOQA
        >>> print(_coerce_mem(30602))
        >>> print(_coerce_mem('4GB'))
        >>> print(_coerce_mem('32GB'))
        >>> print(_coerce_mem('300000000 bytes'))
    """
    if isinstance(mem, int):
        assert mem > 0
    elif isinstance(mem, str):
        import pint
        reg = pint.UnitRegistry()
        mem = reg.parse_expression(mem)
        mem = int(mem.to('megabytes').m)
    else:
        raise TypeError(type(mem))
    return mem


class SlurmJob(cmd_queue.Job):
    """
    Represents a slurm job that hasn't been submitted yet

    Example:
        >>> from watch.utils.slurm_queue import *  # NOQA
        >>> self = SlurmJob('python -c print("hello world")', 'hi', cpus=5, gpus=1, mem='10GB')
        >>> command = self._build_sbatch_args()
        >>> print('command = {!r}'.format(command))
        >>> self = SlurmJob('python -c print("hello world")', 'hi', cpus=5, gpus=1, mem='10GB', depends=[self])
        >>> command = self._build_command()
        >>> print(command)

    """
    def __init__(self, command, name=None, output_fpath=None, depends=None,
                 partition=None, cpus=None, gpus=None, mem=None, begin=None,
                 shell=None):
        if name is None:
            import uuid
            name = 'job-' + str(uuid.uuid4())
        if depends is not None and not ub.iterable(depends):
            depends = [depends]
        self.command = command
        self.name = name
        self.output_fpath = output_fpath
        self.depends = depends
        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.begin = begin
        self.shell = shell
        # if shell not in {None, 'bash'}:
        #     raise NotImplementedError(shell)

        self.jobid = None  # only set once this is run (maybe)
        # --partition=community --cpus-per-task=5 --mem=30602 --gres=gpu:1

    def __nice__(self):
        return repr(self.command)

    def _build_command(self):
        return ' '.join(self._build_sbatch_args())

    def _build_sbatch_args(self, jobname_to_varname=None):
        # job_name = 'todo'
        # output_fpath = '$HOME/.cache/slurm/logs/job-%j-%x.out'
        # command = "python -c 'import sys; sys.exit(1)'"
        # -c 2 -p priority --gres=gpu:1
        sbatch_args = ['sbatch']
        if self.name:
            sbatch_args.append(f'--job-name="{self.name}"')
        if self.cpus:
            sbatch_args.append(f'--cpus-per-task={self.cpus}')
        if self.mem:
            mem = _coerce_mem(self.mem)
            sbatch_args.append(f'--mem={mem}')
        if self.gpus:
            def _coerce_gres(gpus):
                if isinstance(gpus, str):
                    gres = gpus
                elif isinstance(gpus, int):
                    gres = f'gpu:{gpus}'
                else:
                    raise TypeError(type(self.gpus))
                return gres
            gres = _coerce_gres(self.gpus)
            sbatch_args.append(f'--gres="{gres}"')
        if self.output_fpath:
            sbatch_args.append(f'--output="{self.output_fpath}"')

        import shlex
        wrp_command = shlex.quote(self.command)

        if self.shell:
            wrp_command = shlex.quote(self.shell + ' -c ' + wrp_command)

        sbatch_args.append(f'--wrap {wrp_command}')

        if self.depends:
            # TODO: other depends parts
            type_to_dependencies = {
                'afterok': [],
            }
            depends = self.depends if ub.iterable(self.depends) else [self.depends]

            for item in depends:
                if isinstance(item, SlurmJob):
                    jobid = item.jobid
                    if jobid is None and item.name:
                        if jobname_to_varname and item.name in jobname_to_varname:
                            jobid = '${%s}' % jobname_to_varname[item.name]
                        else:
                            jobid = f"$(squeue --noheader --format %i --name '{item.name}')"
                    type_to_dependencies['afterok'].append(jobid)
                else:
                    # if isinstance(item, int):
                    #     type_to_dependencies['afterok'].append(item)
                    # elif isinstance(item, str):
                    #     name = item
                    #     item = f"$(squeue --noheader --format %i --name '{name}')"
                    #     type_to_dependencies['afterok'].append(item)
                    # else:
                    raise TypeError(type(item))

            # squeue --noheader --format %i --name <JOB_NAME>
            depends_parts = []
            for type_, jobids in type_to_dependencies.items():
                if jobids:
                    part = ':'.join([str(j) for j in jobids])
                    depends_parts.append(f'{type_}:{part}')
            depends_part = ','.join(depends_parts)
            sbatch_args.append(f'"--dependency={depends_part}"')

        if self.begin:
            if isinstance(self.begin, int):
                sbatch_args.append(f'"--begin=now+{self.begin}"')
            else:
                sbatch_args.append(f'"--begin={self.begin}"')
        return sbatch_args


class SlurmQueue(cmd_queue.Queue):
    """
    CommandLine:
       xdoctest -m watch.utils.slurm_queue SlurmQueue

    Example:
        >>> from watch.utils.slurm_queue import *  # NOQA
        >>> self = SlurmQueue()
        >>> job0 = self.submit('echo "hi from $SLURM_JOBID"', begin=0)
        >>> job1 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job0])
        >>> job2 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job1])
        >>> job3 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job2])
        >>> job4 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job3])
        >>> job5 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job4])
        >>> job6 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job0])
        >>> job7 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job5, job6])
        >>> self.write()
        >>> self.rprint()
        >>> # xdoctest: +REQUIRES(--run)
        >>> self.run()
        >>> #if ub.find_exe('slurm'):
        >>> #    self.run()

    Example:
        >>> from watch.utils.slurm_queue import *  # NOQA
        >>> self = SlurmQueue(shell='/bin/bash')
        >>> self.add_header_command('export FOO=bar')
        >>> job0 = self.submit('echo "$FOO"')
        >>> job1 = self.submit('echo "$FOO"', depends=job0)
        >>> job2 = self.submit('echo "$FOO"')
        >>> job3 = self.submit('echo "$FOO"', depends=job2)
        >>> self.sync()
        >>> job4 = self.submit('echo "$FOO"')
        >>> self.sync()
        >>> job5 = self.submit('echo "$FOO"')
        >>> self.rprint()
    """
    def __init__(self, name=None, shell=None):
        import uuid
        import time
        self.jobs = []
        if name is None:
            name = 'SQ'
        stamp = time.strftime('%Y%m%dT%H%M%S')
        self.queue_id = name + '-' + stamp + '-' + ub.hash_data(uuid.uuid4())[0:8]
        self.dpath = ub.Path.appdir('slurm_queue') / self.queue_id
        self.log_dpath = self.dpath / 'logs'
        self.fpath = self.dpath / (self.queue_id + '.sh')
        self.shell = shell
        self.header_commands = []
        self.all_depends = None

    def __nice__(self):
        return self.name

    def write(self):
        import os
        import stat
        text = self.finalize_text()
        self.fpath.parent.ensuredir()
        with open(self.fpath, 'w') as file:
            file.write(text)
        os.chmod(self.fpath, (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR |
            stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP))
        return self.fpath

    def submit(self, command, **kwargs):
        name = kwargs.get('name', None)
        if name is None:
            name = kwargs['name'] = f'J{len(self.jobs):04d}-{self.queue_id}'
            # + '-job-{}'.format(len(self.jobs))
        if 'output_fpath' not in kwargs:
            kwargs['output_fpath'] = self.log_dpath / (name + '.sh')
        if self.shell is not None:
            kwargs['shell'] = kwargs.get('shell', self.shell)
        if self.all_depends:
            depends = kwargs.get('depends', None)
            if depends is None:
                depends = self.all_depends
            else:
                if not ub.iterable(depends):
                    depends = [depends]
                depends = self.all_depends + depends
            kwargs['depends'] = depends

        job = SlurmJob(command, **kwargs)
        self.jobs.append(job)
        return job

    def add_header_command(self, command):
        self.header_commands.append(command)

    def sync(self):
        """
        Mark that all future jobs will depend on the current sink jobs
        """
        graph = self._dependency_graph()
        # Find the jobs that nobody depends on
        sink_jobs = [graph.nodes[n]['job'] for n, d in graph.out_degree if d == 0]
        # All new jobs must depend on these jobs
        self.all_depends = sink_jobs

    def order_jobs(self):
        import networkx as nx
        graph = self._dependency_graph()
        if 0:
            print(nx.forest_str(nx.minimum_spanning_arborescence(graph)))
        new_order = []
        for node in nx.topological_sort(graph):
            job = graph.nodes[node]['job']
            new_order.append(job)
        return new_order

    def finalize_text(self):
        new_order = self.order_jobs()
        commands = []
        homevar = '$HOME'
        commands.append(f'mkdir -p "{self.log_dpath.shrinkuser(homevar)}"')
        jobname_to_varname = {}
        for job in new_order:
            args = job._build_sbatch_args(jobname_to_varname)
            command = ' '.join(args)
            if self.header_commands:
                command = ' && '.join(self.header_commands + [command])
            if 1:
                varname = 'JOB_{:03d}'.format(len(jobname_to_varname))
                command = f'{varname}=$({command} --parsable)'
                jobname_to_varname[job.name] = varname
            commands.append(command)
        text = '\n'.join(commands)
        return text

    def run(self, block=False):
        if not ub.find_exe('sbatch'):
            raise Exception('sbatch not found')
        self.log_dpath.ensuredir()
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
        import io
        import pandas as pd
        jobid_history = set()

        num_at_start = None

        def update_status_table():
            nonlocal num_at_start
            # https://rich.readthedocs.io/en/stable/live.html
            info = ub.cmd('squeue --format="%i %P %j %u %t %M %D %R"')
            stream = io.StringIO(info['out'])
            df = pd.read_csv(stream, sep=' ')
            jobid_history.update(df['JOBID'])

            num_running = (df['ST'] == 'R').sum()
            num_in_queue = len(df)
            total_monitored = len(jobid_history)

            if num_at_start is None:
                num_at_start = len(df)

            table = Table(*['num_running', 'num_in_queue', 'total_monitored', 'num_at_start'],
                          title='slurm-monitor')

            # TODO: determine if slurm has accounting on, and if we can
            # figure out how many jobs errored / passed

            table.add_row(
                f'{num_running}',
                f'{num_in_queue}',
                f'{total_monitored}',
                f'{num_at_start}',
            )

            finished = (num_in_queue == 0)
            return table, finished

        table, finished = update_status_table()
        refresh_rate = 0.4
        with Live(table, refresh_per_second=4) as live:
            while not finished:
                time.sleep(refresh_rate)
                table, finished = update_status_table()
                live.update(table)

    def rprint(self, with_status=False, with_rich=0):
        """
        Print info about the commands, optionally with rich
        """
        # from rich.panel import Panel
        # from rich.syntax import Syntax
        # from rich.console import Console
        # console = Console()
        code = self.finalize_text()
        print(ub.highlight_code(f'# --- {str(self.fpath)}', 'bash'))
        print(ub.highlight_code(code, 'bash'))
        # console.print(Panel(Syntax(code, 'bash'), title=str(self.fpath)))


SLURM_NOTES = """
This shows a few things you can do with slurm

# Queue a job in the background
mkdir -p "$HOME/.cache/slurm/logs"
sbatch --job-name="test_job1" --output="$HOME/.cache/slurm/logs/job-%j-%x.out" --wrap="python -c 'import sys; sys.exit(1)'"
sbatch --job-name="test_job2" --output="$HOME/.cache/slurm/logs/job-%j-%x.out" --wrap="echo 'hello'"

#ls $HOME/.cache/slurm/logs
cat "$HOME/.cache/slurm/logs/test_echo.log"

# Queue a job (and block until completed)
srun -c 2 -p priority --gres=gpu:1 echo "hello"
srun echo "hello"

# List jobs in the queue
squeue
squeue --format="%i %P %j %u %t %M %D %R"

# Show job with specific id (e.g. 6)
scontrol show job 6

# Cancel a job with a specific id
scancel 6

# Cancel all jobs from a user
scancel --user="$USER"

# You can setup complicated pipelines
# https://hpc.nih.gov/docs/job_dependencies.html

# Look at finished jobs
# https://ubccr.freshdesk.com/support/solutions/articles/5000686909-how-to-retrieve-job-history-and-accounting

# Jobs within since 3:30pm
sudo sacct --starttime 15:35:00

sudo sacct
sudo sacct --format="JobID,JobName%30,Partition,Account,AllocCPUS,State,ExitCode,elapsed,start"
sudo sacct --format="JobID,JobName%30,State,ExitCode,elapsed,start"


# SHOW ALL JOBS that ran within MinJobAge
scontrol show jobs


# State of each partitions
sinfo

# If the states of the partitions are in drain, find out the reason
sinfo -R

# For "Low socket*core*thre" FIGURE THIS OUT

# Undrain all nodes, first cancel all jobs
# https://stackoverflow.com/questions/29535118/how-to-undrain-slurm-nodes-in-drain-state
scancel --user="$USER"
scancel --state=PENDING
scancel --state=RUNNING
scancel --state=SUSPENDED

sudo scontrol update nodename=namek state=idle



# How to submit a batch job with a dependency

    sbatch --dependency=<type:job_id[:job_id][,type:job_id[:job_id]]> ...

    Dependency types:

    after:jobid[:jobid...]	job can begin after the specified jobs have started
    afterany:jobid[:jobid...]	job can begin after the specified jobs have terminated
    afternotok:jobid[:jobid...]	job can begin after the specified jobs have failed
    afterok:jobid[:jobid...]	job can begin after the specified jobs have run to completion with an exit code of zero (see the user guide for caveats).
    singleton	jobs can begin execution after all previously launched jobs with the same name and user have ended. This is useful to collate results of a swarm or to send a notification at the end of a swarm.



"""

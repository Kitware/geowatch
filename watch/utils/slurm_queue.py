"""
Work in progress. The idea is to provide a TMUX queue and a SLURM queue that
provide a common high level API, even though functionality might diverge, the
core functionality of running processes asynchronously should be provided.

Notes:
    # Installing and configuring SLURM
    See git@github.com:Erotemic/local.git init/setup_slurm.sh
    Or ~/local/init/setup_slurm.sh in my local checkout

Example:
    >>> import sys, ubelt
    >>> sys.path.append(ubelt.expandpath('~/code/watch'))
    >>> from watch.utils.slurm_queue import *  # NOQA
    >>> dpath = ub.Path.appdir('slurm_queue/tests')
    >>> queue = SlurmQueue()
    >>> job0 = queue.submit(f'echo "here we go"', name='root job')
    >>> job1 = queue.submit(f'mkdir {dpath}', depends=[job0])
    >>> job2 = queue.submit(f'echo "result=42" > {dpath}/test.txt ', depends=[job1])
    >>> job3 = queue.submit(f'cat {dpath}/test.txt', depends=[job2])
    >>> print(queue.finalize_text())
"""
import ubelt as ub


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


class SlurmJob(ub.NiceRepr):
    """
    Represents a slurm job that hasn't been submitted yet

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch'))
        >>> from watch.utils.slurm_queue import *  # NOQA
        >>> from watch.utils.slurm_queue import _coerce_mem
        >>> self = SlurmJob('python -c print("hello world")', 'hi', cpus=5, gpus=1, mem='10GB')
        >>> command = self._build_sbatch_args()
        >>> print('command = {!r}'.format(command))
        >>> self = SlurmJob('python -c print("hello world")', 'hi', cpus=5, gpus=1, mem='10GB', depends=['job2', 3, self])
        >>> command = self._build_command()
        >>> print(command)

    """
    def __init__(self, command, name=None, output_fpath=None, depends=None,
                 partition=None, cpus=None, gpus=None, mem=None, begin=None):
        self.command = command
        if name is None:
            import uuid
            name = 'job-' + str(uuid.uuid4())
        self.name = name
        self.output_fpath = output_fpath
        self.depends = depends
        self.cpus = cpus
        self.gpus = gpus
        self.mem = mem
        self.begin = begin

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
                    gres = f'gres:{gpus}'
                else:
                    raise TypeError(type(self.gpus))
                return gres
            gres = _coerce_gres(self.gpus)
            sbatch_args.append(f'--gres="{gres}"')
        if self.output_fpath:
            sbatch_args.append(f'--output="{self.output_fpath}"')

        import shlex
        wrp_command = shlex.quote(self.command)
        sbatch_args.append(f'--wrap {wrp_command}')

        if self.depends:
            # TODO: other depends parts
            type_to_dependencies = {
                'after': [],
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
                    type_to_dependencies['after'].append(jobid)
                else:
                    # if isinstance(item, int):
                    #     type_to_dependencies['after'].append(item)
                    # elif isinstance(item, str):
                    #     name = item
                    #     item = f"$(squeue --noheader --format %i --name '{name}')"
                    #     type_to_dependencies['after'].append(item)
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


class SlurmQueue:
    """
    Example:
        >>> from watch.utils.slurm_queue import *  # NOQA
        >>> self = SlurmQueue()
        >>> job0 = self.submit('echo "hi from $SLURM_JOBID"', begin=5)
        >>> job1 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job0])
        >>> job2 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job1])
        >>> job3 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job2])
        >>> job4 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job3])
        >>> job5 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job4])
        >>> job6 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job0])
        >>> job7 = self.submit('echo "hi from $SLURM_JOBID"', depends=[job5, job6])
        >>> self.write()
        >>> self.rprint()
        >>> if ub.find_exe('slurm'):
        >>>     self.run()
    """
    def __init__(self):
        import uuid
        self.jobs = []
        self.name = 'queue-' + ub.hash_data(uuid.uuid4())[0:8]
        self.dpath = ub.Path.appdir('slurm_queue')
        self.fpath = self.dpath / (self.name + '.sh')

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

    def order_jobs(self):
        import networkx as nx
        graph = nx.DiGraph()
        # TODO: crawl dependencies too?
        for job in self.jobs:
            graph.add_node(job.name, job=job)
            if job.depends:
                for dep in job.depends:
                    graph.add_edge(dep.name, job.name)
        if 0:
            print(nx.forest_str(nx.minimum_spanning_arborescence(graph)))
        new_order = []
        for node in nx.topological_sort(graph):
            job = graph.nodes[node]['job']
            new_order.append(job)
        return new_order

    def submit(self, command, **kwargs):
        name = kwargs.get('name', None)
        if name is None:
            kwargs['name'] = self.name + '-job-{}'.format(len(self.jobs))
        job = SlurmJob(command, **kwargs)
        self.jobs.append(job)
        return job

    def finalize_text(self):
        new_order = self.order_jobs()
        commands = []
        jobname_to_varname = {}
        for job in new_order:
            args = job._build_sbatch_args(jobname_to_varname)
            command = ' '.join(args)
            if 1:
                varname = 'JOB_{:03d}'.format(len(jobname_to_varname))
                command = f'{varname}=$({command} --parsable)'
                jobname_to_varname[job.name] = varname
            commands.append(command)
        text = '\n'.join(commands)
        return text

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
        # ub.cmd('watch squeue')

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

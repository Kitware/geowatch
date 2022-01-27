"""
Work in progress. The idea is to provide a TMUX queue and a SLURM queue that
provide a common high level API, even though functionality might diverge, the
core functionality of running processes asynchronously should be provided.

Notes:
    # Installing and configuring SLURM
    See git@github.com:Erotemic/local.git init/setup_slurm.sh
    Or ~/local/init/setup_slurm.sh in my local checkout
"""
import ubelt as ub


class SlurmQueue:
    def submit(self, command):
        pass

    def _slurm_submit_job():
        job_name = 'todo'
        output_fpath = '$HOME/.cache/slurm/logs/job-%j-%x.out'
        command = "python -c 'import sys; sys.exit(1)'"

        # -c 2 -p priority --gres=gpu:1

        sbatch_command = [
            'sbatch',
            f'--job-name="{job_name}"'
            f'--output="{output_fpath}"'
            f'--warp="{command}"'
        ]
        ub.cmd(sbatch_command)


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

"""

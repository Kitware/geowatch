"""
References:
    https://jmmv.dev/2018/03/shell-readability-strict-mode.html
    https://stackoverflow.com/questions/13195655/bash-set-x-without-it-being-printed
"""
import ubelt as ub
import stat
import os
import uuid


from watch.utils import cmd_queue  # NOQA


class BashJob(cmd_queue.Job):
    """
    A job meant to run inside of a larger bash file. Analog of SlurmJob

    Example:
        >>> from watch.utils.serial_queue import *  # NOQA
        >>> self = BashJob('echo hi', 'myjob')
        >>> self.rprint(1, 1)
    """
    def __init__(self, command, name=None, depends=None, gpus=None, cpus=None,
                 mem=None, quiet=0, info_dpath=None, **kwargs):
        if depends is not None and not ub.iterable(depends):
            depends = [depends]
        self.name = name
        self.kwargs = kwargs  # unused kwargs
        self.command = command
        self.depends = depends
        self.quiet = quiet
        if info_dpath is None:
            info_dpath = ub.Path.appdir('cmd_queue/rando_jobs/') / self.name
        self.info_dpath = info_dpath
        self.pass_fpath = self.info_dpath / f'passed/{self.name}.pass'
        self.fail_fpath = self.info_dpath / f'failed/{self.name}.fail'
        self.stat_fpath = self.info_dpath / f'status/{self.name}.stat'

    def finalize_text(self, with_status=True, with_gaurds=True):
        script = []

        if with_status:
            if self.depends:
                # Dont allow us to run if any dependencies have failed
                conditions = []
                for dep in self.depends:
                    conditions.append(f'[ -f {dep.pass_fpath} ]')
                condition = ' || '.join(conditions)
                script.append(f'if {condition}; then')

        if with_gaurds and not self.quiet:
            script.append('set +e')  # allow the command to fail
            # Tells bash to print the command before it executes it
            script.append('set -x')

        script.append(self.command)

        if with_gaurds:
            # Tells bash to stop printing commands, but
            # is clever in that it cpatures the last return code
            # and doesnt print this command.
            script.append('{ RETURN_CODE=$? ; set +x; } 2>/dev/null')
            # Dont let our boilerplate fail
            script.append('set -e')
            # Old methods:
            # script.append('set +x')
            # script.append('{ set +x; } 2>/dev/null')
        else:
            if with_status:
                script.append('RETURN_CODE=$?')

        if with_status:
            # import shlex
            json_fmt_parts = [
                ('ret', '%s', '$RETURN_CODE'),
                ('name', '"%s"', self.name),
                # ('command', '"%s"', shlex.quote(self.command)),
            ]
            dump_status = _bash_json_dump(json_fmt_parts, self.stat_fpath)
            script.append(f'mkdir -p {self.stat_fpath.parent}')
            script.append(dump_status)
            script.append(ub.codeblock(
                f'''
                if [[ "$RETURN_CODE" == "0" ]]; then
                    mkdir -p {self.pass_fpath.parent}
                    printf "pass" > {self.pass_fpath}
                else
                    mkdir -p {self.fail_fpath.parent}
                    printf "fail" > {self.fail_fpath}
                fi
                '''))

        if with_status:
            if self.depends:
                script.append('else')
                script.append('RETURN_CODE=126')
                script.append('fi')

        text = '\n'.join(script)
        return text

    def rprint(self, with_status=False, with_gaurds=False, with_rich=0):
        """
        Print info about the commands, optionally with rich

        Example:
            >>> from watch.utils.serial_queue import *  # NOQA
            >>> self = SerialQueue('test-serial-queue')
            >>> self.submit('echo hi 1')
            >>> self.submit('echo hi 2')
            >>> self.rprint(with_status=True)
            >>> print('\n\n---\n\n')
            >>> self.rprint(with_status=0)
        """
        code = self.finalize_text(with_status=with_status,
                                  with_gaurds=with_gaurds)
        if with_rich:
            from rich.syntax import Syntax
            from rich.console import Console
            console = Console()
            console.print(Syntax(code, 'bash'))
        else:
            print(ub.highlight_code(code, 'bash'))


class SerialQueue(ub.NiceRepr):
    r"""
    A linear job queue written to a single bash file

    TODO:
        Change this name to just be a Command Script.

        This should be the analog of ub.cmd.

        Using ub.cmd is for one command.
        Using ub.Script is for multiple commands

    Example:
        >>> from watch.utils.serial_queue import *  # NOQA
        >>> self = SerialQueue('test-serial-queue', rootid='test-serial')
        >>> job1 = self.submit('echo hi 1 && false')
        >>> job2 = self.submit('echo hi 2 && true')
        >>> job3 = self.submit('echo hi 3 && true', depends=job1)
        >>> self.rprint()
        >>> self.run()
        >>> self.read_state()
    """
    def __init__(self, name='', dpath=None, rootid=None, environ=None, cwd=None):
        if rootid is None:
            rootid = str(ub.timestamp()) + '_' + ub.hash_data(uuid.uuid4())[0:8]
        self.name = name
        self.rootid = rootid
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('cmd_queue', self.pathid)
        self.dpath = ub.Path(dpath)

        self.fpath = self.dpath / (self.pathid + '.sh')
        self.state_fpath = self.dpath / 'serial_queue_state_{}.txt'.format(self.pathid)
        self.environ = environ
        self.header = '#!/bin/bash'
        self.header_commands = []
        self.jobs = []
        self.cwd = cwd
        self.job_info_dpath = self.dpath / 'job_info'

    def __len__(self):
        return len(self.jobs)

    @property
    def pathid(self):
        """ A path-safe identifier for file names """
        return '{}_{}'.format(self.name, self.rootid)

    def __nice__(self):
        return f'{self.pathid} - {len(self.jobs)}'

    def finalize_text(self, with_status=True, with_gaurds=True):
        script = [self.header]

        total = len(self.jobs)

        if with_gaurds:
            script.append('set +e')

        if with_status:
            script.append(ub.codeblock(
                f'''
                # Init state to keep track of job progress
                (( "_QUEUE_NUM_ERRORED=0" ))
                (( "_QUEUE_NUM_FINISHED=0" ))
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

                # Name, format-string, and value for json status
                json_fmt_parts = [
                    ('status', '"%s"', '$_QUEUE_STATUS'),
                    ('finished', '%d', '$_QUEUE_NUM_FINISHED'),
                    ('errored', '%d', '$_QUEUE_NUM_ERRORED'),
                    ('total', '%d', '$_QUEUE_TOTAL'),
                    ('name', '"%s"', self.name),
                    ('rootid', '"%s"', self.rootid),
                ]
                dump_code = _bash_json_dump(json_fmt_parts, self.state_fpath)
                script.append(dump_code)
                # script.append('cat ' + str(self.state_fpath))

        def _command_enter():
            if with_gaurds:
                # Tells bash to print the command before it executes it
                script.append('set -e')
                script.append('set -x')

        def _command_exit():
            if with_gaurds:
                script.append('{ RETURN_CODE=$? ; set +x; } 2>/dev/null')
                script.append('set +e')
            else:
                if with_status:
                    script.append('RETURN_CODE=$?')

        _mark_status('init')
        if self.environ:
            _mark_status('set_environ')
            if with_gaurds:
                _command_enter()
            script.extend([
                f'export {k}="{v}"' for k, v in self.environ.items()])
            if with_gaurds:
                _command_exit()

        if self.cwd:
            script.append(f'cd {self.cwd}')

        for command in self.header_commands:
            _command_enter()
            script.append(command)
            _command_exit()

        for num, job in enumerate(self.jobs):
            _mark_status('run')
            script.append(ub.codeblock(
                '''
                #
                # Command {} / {} - {}
                ''').format(num + 1, total, job.name))
            script.append(job.finalize_text(with_status, with_gaurds))

            if with_status:
                # Check command status and update the bash state
                script.append(ub.codeblock(
                    '''
                    if [[ "$RETURN_CODE" == "0" ]]; then
                        (( "_QUEUE_NUM_FINISHED=_QUEUE_NUM_FINISHED+1" ))
                    else
                        (( "_QUEUE_NUM_ERRORED=_QUEUE_NUM_ERRORED+1" ))
                    fi
                    '''))
        if with_gaurds:
            script.append('set -e')

        _mark_status('done')
        text = '\n'.join(script)
        return text

    def add_header_command(self, command):
        self.header_commands.append(command)

    def submit(self, command, **kwargs):
        # TODO: we could accept additional args here that modify how we handle
        # the command in the bash script we build (i.e. if the script is
        # allowed to fail or not)
        # self.commands.append(command)
        if isinstance(command, str):
            name = kwargs.get('name', None)
            if name is None:
                name = kwargs['name'] = self.name + '-job-{}'.format(len(self.jobs))
            job = BashJob(command, **kwargs)
        else:
            # Assume job is already a bash job
            job = command
        self.jobs.append(job)
        return job

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

        Example:
            >>> from watch.utils.serial_queue import *  # NOQA
            >>> self = SerialQueue('test-serial-queue')
            >>> self.submit('echo hi 1')
            >>> self.submit('echo hi 2')
            >>> self.rprint(with_status=True)
            >>> print('\n\n---\n\n')
            >>> self.rprint(with_status=0)
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

    def run(self):
        self.write()
        # os.system(f'bash {self.fpath}')
        # verbose=3, check=True)
        # ub.cmd(f'bash {self.fpath}', verbose=3, check=True, system=True)
        ub.cmd(f'bash {self.fpath}', verbose=3, check=True, shell=True)

    def read_state(self):
        import json
        import time
        max_attempts = 100
        num_attempts = 0
        while True:
            try:
                state = json.loads(self.state_fpath.read_text())
            except FileNotFoundError:
                state = {
                    'name': self.name,
                    'status': 'unknown',
                    'total': len(self.jobs),
                    'finished': None,
                    'errored': None,
                }
            except json.JSONDecodeError:
                # we might have tried to read the file while it was being
                # written try again.
                num_attempts += 1
                if num_attempts > max_attempts:
                    raise
                time.sleep(0.01)
                continue
            break
        return state


def _bash_json_dump(json_fmt_parts, fpath):
    printf_body_parts = [
        '"{}": {}'.format(k, f) for k, f, v in json_fmt_parts
    ]
    printf_arg_parts = [
        '"{}"'.format(v) for k, f, v in json_fmt_parts
    ]
    printf_body = '\'{' + ', '.join(printf_body_parts) + '}\\n\''
    printf_args = ' '.join(printf_arg_parts)
    redirect_part = '> ' + str(fpath)
    printf_part = 'printf ' +  printf_body + '\\\n    ' + printf_args
    dump_code = printf_part + ' \\\n    ' + redirect_part
    return dump_code

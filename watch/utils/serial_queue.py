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


def indent(text, prefix='    '):
    r"""
    Indents a block of text

    Args:
        text (str): text to indent
        prefix (str, default = '    '): prefix to add to each line

    Returns:
        str: indented text

        >>> from watch.utils.serial_queue import *  # NOQA
        >>> text = ['aaaa', 'bb', 'cc\n   dddd\n    ef\n']
        >>> text = indent(text)
        >>> print(text)
        >>> text = indent(text)
        >>> print(text)
    """
    if isinstance(text, (list, tuple)):
        return indent('\n'.join(text), prefix)
    else:
        return prefix + text.replace('\n', '\n' + prefix)


class BashJob(cmd_queue.Job):
    r"""
    A job meant to run inside of a larger bash file. Analog of SlurmJob

    Example:
        >>> from watch.utils.serial_queue import *  # NOQA
        >>> self = BashJob('echo hi', 'myjob')
        >>> self.rprint(0, 1)
        >>> self.rprint(1, 1)
    """
    def __init__(self, command, name=None, depends=None, gpus=None, cpus=None,
                 mem=None, bookkeeper=0, info_dpath=None, **kwargs):
        if depends is not None and not ub.iterable(depends):
            depends = [depends]
        self.name = name
        self.pathid = self.name + '_' + ub.hash_data(uuid.uuid4())[0:8]
        self.kwargs = kwargs  # unused kwargs
        self.command = command
        self.depends = depends
        self.bookkeeper = bookkeeper
        if info_dpath is None:
            info_dpath = ub.Path.appdir('cmd_queue/jobinfos/') / self.pathid
        self.info_dpath = info_dpath
        self.pass_fpath = self.info_dpath / f'passed/{self.pathid}.pass'
        self.fail_fpath = self.info_dpath / f'failed/{self.pathid}.fail'
        self.stat_fpath = self.info_dpath / f'status/{self.pathid}.stat'

    def finalize_text(self, with_status=True, with_gaurds=True, conditionals=None):
        script = []
        prefix_script = []
        suffix_script = []

        if with_status:
            if self.depends:
                # Dont allow us to run if any dependencies have failed
                conditions = []
                for dep in self.depends:
                    conditions.append(f'[ -f {dep.pass_fpath} ]')
                condition = ' || '.join(conditions)
                prefix_script.append(f'if {condition}; then')

        if with_gaurds and not self.bookkeeper:
            # -x Tells bash to print the command before it executes it
            script.append('set +e -x')  # and +e allow the command to fail

        script.append(self.command)

        if with_gaurds:
            # Tells bash to stop printing commands, but
            # is clever in that it cpatures the last return code
            # and doesnt print this command.
            # Also set -e so our boilerplate is not allowed to fail
            script.append('{ RETURN_CODE=$? ; set +x -e; } 2>/dev/null')
        else:
            if with_status:
                script.append('RETURN_CODE=$?')

        if with_status:
            if self.depends:
                suffix_script.append('else')
                suffix_script.append('    RETURN_CODE=126')
                suffix_script.append('fi')
                script = prefix_script + [indent(script)] + suffix_script

        if with_status:
            # import shlex
            json_fmt_parts = [
                ('ret', '%s', '$RETURN_CODE'),
                ('name', '"%s"', self.name),
                # ('command', '"%s"', shlex.quote(self.command)),
            ]
            dump_status = _bash_json_dump(json_fmt_parts, self.stat_fpath)

            _job_conditionals = {
                'on_pass': [
                    f'mkdir -p {self.pass_fpath.parent}',
                    f'printf "pass" > {self.pass_fpath}',
                ],
                'on_fail': [
                    f'mkdir -p {self.fail_fpath.parent}',
                    f'printf "fail" > {self.fail_fpath}',
                ]
            }

            if conditionals:
                for k, v in _job_conditionals.items():
                    if k in conditionals:
                        v2 = conditionals.get(k)
                        if not ub.iterable(v2):
                            v2 = [v2]
                        v.extend(v2)

            on_pass_part = indent(_job_conditionals['on_pass'])
            on_fail_part = indent(_job_conditionals['on_fail'])
            conditional_body = '\n'.join([
                'if [[ "$RETURN_CODE" == "0" ]]; then',
                on_pass_part,
                'else',
                on_fail_part,
                'fi'
            ])
            script.append('# <bookkeeping> ')
            script.append(f'mkdir -p {self.stat_fpath.parent}')
            script.append(dump_status)
            script.append(conditional_body)
            script.append('# </bookkeeping> ')

        assert isinstance(script, list)
        text = '\n'.join(script)
        return text

    def rprint(self, with_status=False, with_gaurds=False, with_rich=0):
        r"""
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


class SerialQueue(cmd_queue.Queue):
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
        >>> self.rprint(1, 1)
        >>> self.run()
        >>> self.read_state()
    """
    def __init__(self, name='', dpath=None, rootid=None, environ=None, cwd=None, **kwargs):
        super().__init__()
        if rootid is None:
            rootid = str(ub.timestamp().split('T')[0]) + '_' + ub.hash_data(uuid.uuid4())[0:8]
        self.name = name
        self.rootid = rootid
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('cmd_queue', self.pathid)
        self.dpath = ub.Path(dpath)

        self.unused_kwargs = kwargs

        self.fpath = self.dpath / (self.pathid + '.sh')
        self.state_fpath = self.dpath / 'serial_queue_{}.txt'.format(self.pathid)
        self.environ = environ
        self.header = '#!/bin/bash'
        self.header_commands = []
        self.jobs = []

        self.cwd = cwd
        self.job_info_dpath = self.dpath / 'job_info'

    @property
    def pathid(self):
        """ A path-safe identifier for file names """
        return '{}_{}'.format(self.name, self.rootid)

    def __nice__(self):
        return f'{self.pathid} - {self.num_real_jobs}'

    def finalize_text(self, with_status=True, with_gaurds=True):
        script = [self.header]

        total = self.num_real_jobs

        if with_gaurds:
            script.append('set -e')

        if with_status:
            script.append(ub.codeblock(
                f'''
                # Init state to keep track of job progress
                (( "_CMD_QUEUE_NUM_ERRORED=0" )) || true
                (( "_CMD_QUEUE_NUM_FINISHED=0" )) || true
                _CMD_QUEUE_TOTAL={total}
                _CMD_QUEUE_STATUS=""
                '''))

        old_status = None

        def _mark_status(status):
            nonlocal old_status
            # be careful with json formatting here
            if with_status:
                if old_status != status:
                    script.append(ub.codeblock(
                        '''
                        _CMD_QUEUE_STATUS="{}"
                        ''').format(status))

                old_status = status

                # Name, format-string, and value for json status
                json_fmt_parts = [
                    ('status', '"%s"', '$_CMD_QUEUE_STATUS'),
                    ('finished', '%d', '$_CMD_QUEUE_NUM_FINISHED'),
                    ('errored', '%d', '$_CMD_QUEUE_NUM_ERRORED'),
                    ('total', '%d', '$_CMD_QUEUE_TOTAL'),
                    ('name', '"%s"', self.name),
                    ('rootid', '"%s"', self.rootid),
                ]
                dump_code = _bash_json_dump(json_fmt_parts, self.state_fpath)
                script.append(dump_code)
                # script.append('cat ' + str(self.state_fpath))

        def _command_enter():
            if with_gaurds:
                # Tells bash to print the command before it executes it
                script.append('set -x')

        def _command_exit():
            if with_gaurds:
                script.append('{ set +x; } 2>/dev/null')
            else:
                if with_status:
                    script.append('RETURN_CODE=$?')

        _mark_status('init')
        if self.environ:
            script.append('#')
            script.append('# Environment')
            _mark_status('set_environ')
            if with_gaurds:
                _command_enter()
            script.extend([
                f'export {k}="{v}"' for k, v in self.environ.items()])
            if with_gaurds:
                _command_exit()

        if self.cwd:
            script.append('#')
            script.append('# Working Directory')
            script.append(f'cd {self.cwd}')

        if self.header_commands:
            script.append('#')
            script.append('# Header commands')
            for command in self.header_commands:
                _command_enter()
                script.append(command)
                _command_exit()

        if self.jobs:
            script.append('#')
            script.append('# Jobs')

            num = 0
            for job in self.jobs:
                if job.bookkeeper:
                    if with_status:
                        script.append(job.finalize_text(with_status, with_gaurds))
                else:
                    if with_status:
                        script.append('# <command>')

                    _mark_status('run')

                    script.append(ub.codeblock(
                        '''
                        #
                        ### Command {} / {} - {}
                        ''').format(num + 1, total, job.name))

                    conditionals = {
                        'on_pass': '(( "_CMD_QUEUE_NUM_FINISHED=_CMD_QUEUE_NUM_FINISHED+1" )) || true',
                        'on_fail': '(( "_CMD_QUEUE_NUM_ERRORED=_CMD_QUEUE_NUM_ERRORED+1" )) || true',
                    }
                    script.append(job.finalize_text(with_status, with_gaurds, conditionals))
                    if with_status:
                        script.append('# </command>')
                    num += 1

        if with_gaurds:
            script.append('set +e')

        _mark_status('done')
        text = '\n'.join(script)
        return text

    def add_header_command(self, command):
        self.header_commands.append(command)

    # def sync(self):
    #     pass

    # def submit(self, command, **kwargs):
    #     # TODO: we could accept additional args here that modify how we handle
    #     # the command in the bash script we build (i.e. if the script is
    #     # allowed to fail or not)
    #     # self.commands.append(command)
    #     if isinstance(command, str):
    #         name = kwargs.get('name', None)
    #         if name is None:
    #             name = kwargs['name'] = self.name + '-job-{}'.format(self.num_real_jobs)
    #         job = BashJob(command, **kwargs)
    #     else:
    #         # Assume job is already a bash job
    #         job = command
    #     self.jobs.append(job)

    #     if not job.bookkeeper:
    #         self.num_real_jobs += 1
    #     return job

    def write(self):
        text = self.finalize_text()
        with open(self.fpath, 'w') as file:
            file.write(text)
        os.chmod(self.fpath, (
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IRUSR |
            stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP))
        return self.fpath

    def rprint(self, with_status=False, with_gaurds=False, with_rich=0):
        r"""
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

    def run(self, block=None):
        # block is always true here
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
                    'total': self.num_real_jobs,
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
    printf_body = r"'{" + ", ".join(printf_body_parts) + r"}\n'"
    printf_args = ' '.join(printf_arg_parts)
    redirect_part = '> ' + str(fpath)
    printf_part = 'printf ' +  printf_body + ' \\\n    ' + printf_args
    dump_code = printf_part + ' \\\n    ' + redirect_part
    return dump_code

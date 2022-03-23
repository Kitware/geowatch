import ubelt as ub
import stat
import os
import uuid


from watch.utils import cmd_queue  # NOQA


class BashJob(cmd_queue.Job):
    """
    A job meant to run inside of a larger bash file. Analog of SlurmJob
    """
    def __init__(self, command, name=None, depends=None, gpus=None, cpus=None, begin=None):
        if depends is not None and not ub.iterable(depends):
            depends = [depends]

        self.name = name
        self.command = command
        self.depends = depends


class SerialQueue(ub.NiceRepr):
    """
    A linear job queue written to a single bash file

    TODO:
        Change this name to just be a Command Script.

        This should be the analog of ub.cmd.

        Using ub.cmd is for one command.
        Using ub.Script is for multiple commands

    Example:
        >>> self = SerialQueue('foo', 'foo')
        >>> self.rprint()
    """
    def __init__(self, name='', dpath=None, rootid=None, environ=None, cwd=None):
        if rootid is None:
            rootid = str(ub.timestamp()) + '_' + ub.hash_data(uuid.uuid4())[0:8]
        self.name = name
        self.rootid = rootid
        if dpath is None:
            dpath = ub.ensure_app_cache_dir('tmux_queue', self.pathid)
        self.dpath = ub.Path(dpath)

        self.fpath = self.dpath / (self.pathid + '.sh')
        self.state_fpath = self.dpath / 'job_state_{}.txt'.format(self.pathid)
        self.environ = environ
        self.header = '#!/bin/bash'
        self.header_commands = []
        self.commands = []
        self.cwd = cwd

    @property
    def pathid(self):
        """ A path-safe identifier for file names """
        return '{}_{}'.format(self.name, self.rootid)

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

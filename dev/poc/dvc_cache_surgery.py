#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import scriptconfig as scfg
import ubelt as ub
import argparse


# TODO: will be part of scriptconfig in the future
class RawDescriptionDefaultsHelpFormatter(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter):
    pass


class ModalCLI(object):
    """
    Contains multiple scriptconfig.Config items with corresponding `main`
    functions.
    """

    def __init__(self, description='', sub_clis=[], version=None):
        self.description = description
        self.sub_clis = sub_clis
        self.version = version

    def register(self, cli_cls):
        # Note: the order or registration is how it will appear in the CLI help
        # Hack for older scriptconfig
        if not hasattr(cli_cls, 'default'):
            cli_cls.default = cli_cls.__default__
        self.sub_clis.append(cli_cls)
        return cli_cls

    def _build_subcmd_infos(self):
        cmdinfo_list = []
        for cli_cls in self.sub_clis:
            cmdname = getattr(cli_cls, '__command__', None)
            subconfig = cli_cls()
            parserkw = {}
            __alias__ = getattr(cli_cls, '__alias__', [])
            if __alias__:
                parserkw['aliases']  = __alias__
            parserkw.update(subconfig._parserkw())
            parserkw['help'] = parserkw['description'].split('\n')[0]
            cmdinfo_list.append({
                'cmdname': cmdname,
                'parserkw': parserkw,
                'main_func': cli_cls.main,
                'subconfig': subconfig,
            })
        return cmdinfo_list

    def build_parser(self):
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=RawDescriptionDefaultsHelpFormatter,
        )

        if self.version is not None:
            parser.add_argument('--version', action='store_true',
                                help='show version number and exit')

        # Prepare information to be added to the subparser before it is created
        cmdinfo_list = self._build_subcmd_infos()

        # Build a list of primary command names to display as the valid options
        # for subparsers. This avoids cluttering the screen with all aliases
        # which happens by default.
        command_choices = [d['cmdname'] for d in cmdinfo_list]
        metavar = '{' + ','.join(command_choices) + '}'

        # The subparser is what enables the modal CLI. It will redirect a
        # command to a chosen subparser.
        subparser_group = parser.add_subparsers(
            title='commands', help='specify a command to run', metavar=metavar)

        for cmdinfo in cmdinfo_list:
            # Add a new command to subparser_group
            subparser = subparser_group.add_parser(
                cmdinfo['cmdname'], **cmdinfo['parserkw'])
            subparser = cmdinfo['subconfig'].argparse(subparser)
            subparser.set_defaults(main=cmdinfo['main_func'])
        return parser

    def run(self):
        parser = self.build_parser()

        try:
            import argcomplete
            # Need to run: "$(register-python-argcomplete xdev)"
            # or activate-global-python-argcomplete --dest=-
            # activate-global-python-argcomplete --dest ~/.bash_completion.d
            # To enable this.
        except ImportError:
            argcomplete = None

        if argcomplete is not None:
            argcomplete.autocomplete(parser)

        ns = parser.parse_args()
        kw = ns.__dict__

        if kw.pop('version'):
            print(self.version)
            return 0

        sub_main = kw.pop('main', None)
        if sub_main is None:
            parser.print_help()
            raise ValueError('no command given')
            return 1

        try:
            ret = sub_main(cmdline=False, **kw)
        except Exception as ex:
            print('ERROR ex = {!r}'.format(ex))
            raise
            return 1
        else:
            if ret is None:
                ret = 0
            return ret


modal = ModalCLI(
    description=ub.codeblock(
        '''
        DVC Surgery
        '''),
    version='0.0.1',
)


@modal.register
class CachePurgeCLI(scfg.Config):
    """
    Destroy all files in the DVC cache referenced in the target directory.
    """
    __command__ = 'purge'
    __default__ = dict(
        dpath=scfg.Value('.', position=1, help='input path'),
        workers=scfg.Value(0, help='number of parallel jobs'),
    )

    @classmethod
    def main(cls, cmdline=False, **kwargs):
        from watch.utils import util_progress
        from watch.utils.simple_dvc import SimpleDVC
        config = cls(cmdline=cmdline, data=kwargs)
        dpath = ub.Path(config['dpath'])
        dvc = SimpleDVC.coerce(dpath)

        cache_fpath_iter = find_cached_fpaths(dvc, dpath)

        jobs = ub.JobPool(mode='thread', max_workers=4)
        with jobs:
            pman = util_progress.ProgressManager()
            with pman:
                fpath_iter = pman(cache_fpath_iter, desc='deleting cache')
                for fpath in fpath_iter:
                    jobs.submit(fpath.delete)
                for job in pman(jobs.as_completed(), desc='finish deletes'):
                    try:
                        job.result()
                    except Exception as ex:
                        print(f'ex={ex}')


@modal.register
class CacheCopyCLI(scfg.Config):
    """
    Copy all files reference in the current checkout from one cache to another
    cache.
    """
    __command__ = 'copy'
    __default__ = dict(
        dpath=scfg.Value('.', position=1, help='input path'),
        new_cache_dpath=scfg.Value(None, position=2, help='new cache location'),
        workers=scfg.Value(0, help='number of parallel jobs'),
    )

    @classmethod
    def main(cls, cmdline=False, **kwargs):
        """
        Ignore:
            cmdline = 0
            kwargs = dict(
                dpath='/home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_expt_dvc',
                workers=0,
                new_cache_dpath='/data/dvc-caches/smart_expt_dvc_cache'
            )
            cls = CacheCopyCLI
            ...
        """
        config = cls(cmdline=cmdline, data=kwargs)

        from watch.utils import util_progress
        from watch.utils.simple_dvc import SimpleDVC
        dpath = ub.Path(config['dpath'])
        dvc = SimpleDVC.coerce(dpath)

        old_cache_dpath = dvc.cache_dir
        new_cache_dpath = ub.Path(config['new_cache_dpath'])
        workers = config['workers']

        cache_fpath_iter = find_cached_fpaths(dvc, dpath)

        def copy_job(fpath):
            if fpath.exists():
                cache_rel_path = fpath.relative_to(old_cache_dpath)
                new_fpath = new_cache_dpath / cache_rel_path
                if not new_fpath.exists():
                    new_fpath.parent.ensuredir()
                    fpath.copy(new_fpath)

        jobs = ub.JobPool(mode='thread', max_workers=workers)
        with jobs:
            pman = util_progress.ProgressManager()
            with pman:
                for fpath in pman(cache_fpath_iter, desc='moving cache'):
                    jobs.submit(copy_job, fpath)

                for job in pman(jobs.as_completed(), desc='finish moving'):
                    try:
                        job.result()
                    except Exception as ex:
                        print(f'ex={ex}')


def find_cached_fpaths(dvc, dpath):
    for fpath in dvc.find_sidecar_paths(dpath):
        yield from dvc.resolve_cache_paths(ub.Path(fpath))


# class DVCCacheSurgeryConfig(scfg.DataConfig):
#     action = scfg.Value(None, position=1, help='action to perform.', choices=['purge', 'move'])
#     dst = scfg.Value(None, help='new destination cache for the move command')


# def main(cmdline=1, **kwargs):
#     """
#     Example:
#         >>> # xdoctest: +SKIP
#         >>> cmdline = 0
#         >>> kwargs = dict(dpath='.')
#         >>> main(cmdline=cmdline, **kwargs)
#     """
#     config = DVCCacheSurgeryConfig.cli(cmdline=cmdline, data=kwargs)
#     print('config = ' + ub.urepr(dict(config), nl=1))
#     dpath = ub.Path(config['dpath'])

#     from watch.utils.simple_dvc import SimpleDVC
#     dvc = SimpleDVC.coerce(dpath)

#     if config['action'] == 'purge':
#         purge_dvc_cache(dvc)
#     elif config['action'] == 'move':
#         purge_dvc_cache(dvc)
#     else:
#         raise KeyError(config['action'])


# def move_dvc_cache(dvc):
#     from watch.utils import util_progress

#     def find_cached_fpaths():
#         for fpath in dvc.find_sidecar_paths():
#             yield from dvc.resolve_cache_paths(fpath)

#     jobs = ub.JobPool(mode='thread', max_workers=4)
#     with jobs:
#         pman = util_progress.ProgressManager()
#         with pman:
#             fpath_iter = pman(find_cached_fpaths(), desc='deleting cache')
#             for fpath in fpath_iter:
#                 jobs.submit(fpath.delete)
#             for job in pman(jobs.as_completed(), desc='finish deletes'):
#                 try:
#                     job.result()
#                 except Exception as ex:
#                     print(f'ex={ex}')


# def purge_dvc_cache(dvc):


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/dvc_cache_surgery.py --help
    """
    modal.run()

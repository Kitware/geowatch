#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import scriptconfig as scfg
from scriptconfig.modal import ModalCLI
import ubelt as ub


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

    Example:
        cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
        python ~/code/watch/dev/poc/dvc_cache_surgery.py purge . --workers=0

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
        workers = config['workers']
        dvc = SimpleDVC.coerce(dpath)

        cache_fpath_iter = find_cached_fpaths(dvc, dpath)

        jobs = ub.JobPool(mode='thread', max_workers=workers)
        with jobs:
            pman = util_progress.ProgressManager()
            with pman:
                fpath_iter = pman(cache_fpath_iter, desc='submit delete jobs')
                for fpath in fpath_iter:
                    if fpath.exists():
                        jobs.submit(fpath.delete)
                for job in pman(jobs.as_completed(), desc='collect deletes jobs'):
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

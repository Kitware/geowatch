#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class DVCPurgeConfig(scfg.DataConfig):
    dpath = scfg.Value('.', position=1, help='input path')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict(dpath='.')
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = DVCPurgeConfig.cli(cmdline=cmdline, data=kwargs)
    print('config = ' + ub.urepr(dict(config), nl=1))
    dpath = ub.Path(config['dpath'])

    from watch.utils import util_yaml
    from watch.utils import util_progress
    from watch.utils.simple_dvc import SimpleDVC
    dvc = SimpleDVC.coerce(dpath)

    def find_dvc_sidecar_paths():
        for r, ds, fs in dpath.walk():
            for f in fs:
                if f.endswith('.dvc'):
                    yield r / f

    def resolve_tracked_cache_paths(fpath):
        data = util_yaml.yaml_loads(fpath.read_text())
        for item in data['outs']:
            md5 = item['md5']
            cache_fpath = dvc.cache_dir / md5[0:2] / md5[2:]
            if md5.endswith('.dir') and cache_fpath.exists():
                dir_data = util_yaml.yaml_loads(cache_fpath.read_text())
                for item in dir_data:
                    file_md5 = item['md5']
                    file_cache_fpath = dvc.cache_dir / file_md5[0:2] / file_md5[2:]
                    yield file_cache_fpath
            yield cache_fpath

    def find_cached_fpaths():
        for fpath in find_dvc_sidecar_paths():
            yield from resolve_tracked_cache_paths(fpath)

    jobs = ub.JobPool(mode='thread', max_workers=4)
    with jobs:
        pman = util_progress.ProgressManager()
        with pman:
            fpath_iter = pman(find_cached_fpaths(), desc='deleting cache')
            for fpath in fpath_iter:
                jobs.submit(fpath.delete)
            for job in pman(jobs.as_completed(), desc='finish deletes'):
                try:
                    job.result()
                except Exception as ex:
                    print(f'ex={ex}')


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/purge_dvc_cache.py
        python -m purge_dvc_cache
    """
    main()

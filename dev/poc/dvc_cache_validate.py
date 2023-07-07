#!/usr/bin/env python3
import scriptconfig as scfg
"""
The following describes a DVC Error I ran into.

    The cache files were corrupted, and dvc does not seem to have a way to
    check for this (so I wrote this script).

    But after I removed the corrupted cache files on machine2, I tried to push
    them from machine1 (with a good cache) to machine2, but dvc didn't realize
    that machine1 had files that machine2 was missing. Perhaps this is because
    the missing files were behind a .dir object in the cache?

It would be good to get a MWE for this.
"""

import ubelt as ub


class DvcCacheValidateCLI(scfg.DataConfig):
    """
    Checks for corruption in the dvc cache.
    """
    path = scfg.Value(None, help='path to a dvc repo or file to validate the cache for')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from dvc_cache_validate import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict(path='test_video_3.dvc')
            >>> cls = DvcCacheValidateCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        from watch.utils.simple_dvc import SimpleDVC
        dvc = SimpleDVC.coerce(config.path)

        # list(dvc.find_file_tracker(ub.Path(config.path).absolute()))

        # TODO: better way to list all the cache files associated with a
        # directory or a dvc file.
        cache_files = list(dvc.resolve_cache_paths(ub.Path(config.path).absolute()))

        corrupt_fpaths = []
        valid_fpaths = []
        for fpath in ub.ProgIter(cache_files):
            if not fpath.name.endswith('.dir'):
                md5_hash = ub.hash_file(fpath, hasher='md5')
                prefix, suffix = md5_hash[0:2], md5_hash[2:]
                file_prefix = fpath.parent.name
                file_suffix = fpath.name
                if prefix != file_prefix or suffix != file_suffix:
                    print('CORRUPT FILE:')
                    corrupt_fpaths.append(fpath)
                else:
                    valid_fpaths.append(fpath)

        for p in corrupt_fpaths:
            p.delete()


def find_cached_fpaths(dvc, dpath):
    for fpath in dvc.find_sidecar_paths(dpath):
        yield from dvc.resolve_cache_paths(ub.Path(fpath))


__cli__ = DvcCacheValidateCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/dvc_cache_validate.py
        python -m dvc_cache_validate
    """
    main()

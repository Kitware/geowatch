"""
Continuously watch a directory and cleanup gdal temp files that fail to be
removed.
"""
#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CleanupGdalTmpFileWatcherCLI(scfg.DataConfig):
    dpath = scfg.Value('/data2/projects/smart/smart_phase3_data/Aligned-Drop8-ARA', help='param1')
    age_thresh = scfg.Value('1 minute')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from cleanup_gdal_tmp_file_watcher import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = CleanupGdalTmpFileWatcherCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        import kwutil
        import time
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))
        dpath = ub.Path(config.dpath)

        ureg = kwutil.util_time._time_unit_registry()

        age_threshold = kwutil.util_time.timedelta.coerce(config.age_thresh)

        # 12 * ureg.hours
        wait_time = 10 * ureg.seconds

        while True:
            print('Start walk')
            num_removed = 0
            num_keep = 0
            for root, ds, fs in dpath.walk():
                for fname in fs:
                    if fname.startswith('.tmp'):
                        fpath = root / fname
                        mtime = kwutil.util_time.datetime.coerce(fpath.stat().st_mtime)
                        now_time = kwutil.util_time.datetime.coerce('now')
                        age_delta = kwutil.util_time.timedelta.coerce(now_time - mtime)
                        # age = age_delta.total_seconds() * ureg.seconds
                        if age_delta > age_threshold:
                            print(f'Found old tmp file: {fpath}')
                            num_removed += 1
                            fpath.delete()
                        else:
                            print(f'Found young tmp file: {fpath}')
                            num_keep += 1

            print(f'num_removed={num_removed}')
            print(f'num_keep={num_keep}')
            print('waiting to check again')
            time.sleep(wait_time.to('seconds').m)

__cli__ = CleanupGdalTmpFileWatcherCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/dev/poc/cleanup_gdal_tmp_file_watcher.py \
            --dpath /data2/projects/smart/smart_phase3_data/Aligned-Drop8-ARA \
            --age_thresh "1 hour"
    """
    main()

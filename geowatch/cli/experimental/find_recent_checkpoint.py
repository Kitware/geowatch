#!/usr/bin/env python3
import scriptconfig as scfg


class FindRecentCheckpointCLI(scfg.DataConfig):
    """
    Helper script to lookup the most recent checkpoint.

    Not sure what the best home for this script is. Useful to help make train
    scripts more consise.  Perhaps this is part of the mlops CLI?
    """
    default_root_dir = scfg.Value(None, help='the default root dir passed to lightning', position=1)
    allow_last = scfg.Value(True, isflag=True, help='if True, then prevent the last.ckpt from being chosen')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from geowatch.cli.experimental.find_recent_checkpoint import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = FindRecentCheckpointCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import ubelt as ub
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        root_dir = ub.Path(config.default_root_dir)
        checkpoints = list((root_dir / 'lightning_logs').glob('version_*/checkpoints/*.ckpt'))
        if len(checkpoints) == 0:
            print('None')
        else:
            version_to_checkpoints = ub.group_items(checkpoints, key=lambda x: int(x.parent.parent.name.split('_')[-1]))
            max_version = max(version_to_checkpoints)
            candidates = version_to_checkpoints[max_version]
            if not config.allow_last:
                checkpoints = [p for p in checkpoints if 'last.ckpt' not in p.name]
            checkpoints = sorted(candidates, key=lambda p: p.stat().st_mtime)
            chosen = checkpoints[-1]
            print(chosen)

__cli__ = FindRecentCheckpointCLI
main = __cli__.main

if __name__ == '__main__':
    """
    CommandLine:
        python -m geowatch.cli.experimental.find_recent_checkpoint
    """
    main()

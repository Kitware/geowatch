#!/usr/bin/env python3
import scriptconfig as scfg


class FindRecentCheckpointCLI(scfg.DataConfig):
    """
    Helper script to lookup the most recent checkpoint.

    Not sure what the best home for this script is. Useful to help make train
    scripts more consise.  Perhaps this is part of the mlops CLI?

    Usage
    -----

    This prints out extra arguments to be used at the end of a lightning CLI
    invocation.  As such, you should ensure its contents are read into a bash
    array, and that array should be passed to the invocation such that any
    bash-level word splitting is explicit.

    In short this is the following pattern that should be used

    .. code:: bash

        PREV_CHECKPOINT_TEXT=$(python -m geowatch.cli.experimental.find_recent_checkpoint --default_root_dir="$DEFAULT_ROOT_DIR")
        echo "PREV_CHECKPOINT_TEXT = $PREV_CHECKPOINT_TEXT"
        if [[ "$PREV_CHECKPOINT_TEXT" == "None" ]]; then
            PREV_CHECKPOINT_ARGS=()
        else
            PREV_CHECKPOINT_ARGS=(--ckpt_path "$PREV_CHECKPOINT_TEXT")
        fi
        echo "${PREV_CHECKPOINT_ARGS[@]}"


    This method of usage will do nothing when there is no checkpoint, and add
    the appropriate restart argument when something is needed.
    """
    default_root_dir = scfg.Value(None, help='the default root dir passed to lightning', position=1)
    allow_last = scfg.Value(True, isflag=True, help='if True, then prevent the last.ckpt from being chosen')
    as_cli_arg = scfg.Value(False, isflag=True, help='if True, print text that can be used to extend a lightning CLI invocation')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> import ubelt as ub
            >>> from geowatch.cli.experimental.find_recent_checkpoint import *  # NOQA
            >>> cmdline = 0
            >>> # Make a dummy train directory
            >>> default_root_dir = ub.Path.appdir('geowatch/tests/find_recent_checkpoint')
            >>> fake_dpath = (default_root_dir / 'lightning_logs/version_0/checkpoints').ensuredir()
            >>> (fake_dpath / 'pretend.ckpt').write_text('dummy')
            >>> kwargs = dict(default_root_dir=default_root_dir)
            >>> cls = FindRecentCheckpointCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
            >>> kwargs['as_cli_arg'] = True
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

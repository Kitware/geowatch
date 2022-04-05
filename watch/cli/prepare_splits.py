"""
CommandLine:
    xdoctest -m watch.cli.prepare_splits __doc__

Example:
    >>> from watch.cli.prepare_splits import *  # NOQA
    >>> base_fpath = 'data.kwcoco.json'
    >>> config = {
    >>>     'base_fpath': './bundle/data.kwcoco.json',
    >>>     'virtualenv_cmd': 'conda activate watch',
    >>>     'run': 0,
    >>>     'cache': False,
    >>>     'backend': 'serial',
    >>>     'verbose': 0,
    >>> }
    >>> queue = prep_splits(cmdline=False, **config)
    >>> config['backend'] = 'slurm'
    >>> queue = prep_splits(cmdline=False, **config)
    >>> queue.rprint(0, 0)
    >>> config['backend'] = 'tmux'
    >>> queue = prep_splits(cmdline=False, **config)
    >>> queue.rprint(0, 0)
    >>> config['backend'] = 'serial'
    >>> queue = prep_splits(cmdline=False, **config)
    >>> queue.rprint(0, 0)

"""

import scriptconfig as scfg
import ubelt as ub


class PrepareSplitsConfig(scfg.Config):
    """
    This generates the bash commands necessary to split a base kwcoco file into
    the standard train / validation splits.

    """
    default = {
        'base_fpath': scfg.Value(None, help='base coco file to split'),
        'virtualenv_cmd': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your bashrc
            does not start it by default.''')),
        'keep_sessions': scfg.Value(False, help='if True does not close tmux sessions'),
        'run': scfg.Value(True, help='if True execute the pipeline'),
        'cache': scfg.Value(True, help='if True skip completed results'),
        'serial': scfg.Value(False, help='if True use serial mode'),
        'backend': scfg.Value('tmux', help=None),
        'verbose': scfg.Value(1, help=''),
    }


def _submit_split_jobs(base_fpath, queue, depends=[]):
    """
    Populates a Serial, Slurm, or Tmux Queue with jobs
    """

    base_fpath = ub.Path(base_fpath)

    splits = {
        'nowv': base_fpath.augment(suffix='_nowv', multidot=True),

        'train': base_fpath.augment(suffix='_train', multidot=True),
        'nowv_train': base_fpath.augment(suffix='_nowv_train', multidot=True),
        'wv_train': base_fpath.augment(suffix='_wv_train', multidot=True),
        's2_wv_train': base_fpath.augment(suffix='_s2_wv_train', multidot=True),

        'vali': base_fpath.augment(suffix='_vali', multidot=True),
        'nowv_vali': base_fpath.augment(suffix='_nowv_vali', multidot=True),
        'wv_vali': base_fpath.augment(suffix='_wv_vali', multidot=True),
        's2_wv_vali': base_fpath.augment(suffix='_s2_wv_vali', multidot=True),
    }

    split_jobs = {}
    # Perform train/validation splits with and without worldview
    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {base_fpath} \
            --dst {splits['train']} \
            --select_videos '.name | startswith("KR_") | not'
        ''')
    split_jobs['train'] = queue.submit(command, begin=1, depends=depends)

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['train']} \
            --dst {splits['nowv_train']} \
            --select_images '.sensor_coarse != "WV"'
        ''')
    queue.submit(command, depends=[split_jobs['train']])

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['train']} \
            --dst {splits['wv_train']} \
            --select_images '.sensor_coarse == "WV"'
        ''')
    queue.submit(command, depends=[split_jobs['train']])

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['train']} \
            --dst {splits['s2_wv_train']} \
            --select_images '.sensor_coarse == "WV" or .sensor_coarse == "S2"'
        ''')
    queue.submit(command, depends=[split_jobs['train']])

    # Perform vali/validation splits with and without worldview
    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {base_fpath} \
            --dst {splits['vali']} \
            --select_videos '.name | startswith("KR_")'
        ''')
    split_jobs['vali'] = queue.submit(command, depends=depends)

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['vali']} \
            --dst {splits['nowv_vali']} \
            --select_images '.sensor_coarse != "WV"'
        ''')
    queue.submit(command, depends=[split_jobs['vali']])

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['vali']} \
            --dst {splits['s2_wv_vali']} \
            --select_images '.sensor_coarse == "WV" or .sensor_coarse == "S2"'
        ''')
    queue.submit(command, depends=[split_jobs['vali']])

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['vali']} \
            --dst {splits['wv_vali']} \
            --select_images '.sensor_coarse == "WV"'
        ''')
    queue.submit(command, depends=[split_jobs['vali']])

    # Add in additional no-worldview full dataset
    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {base_fpath} \
            --dst {splits['nowv']} \
            --select_images '.sensor_coarse != "WV"'
        ''')
    queue.submit(command, depends=depends)
    return queue


def prep_splits(cmdline=False, **kwargs):
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.

    TODO:
        - [ ] Option to just dump the serial bash script that does everything.
    """
    config = PrepareSplitsConfig(cmdline=cmdline)
    config.update(kwargs)

    if config['base_fpath'] == 'auto':
        # Auto hack.
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
        # base_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
    else:
        base_fpath = ub.Path(config['base_fpath'])

    # queue = tmux_queue.TMUXMultiQueue(name='watch-splits', size=2)
    from watch.utils import cmd_queue
    queue = cmd_queue.Queue.create(
        backend=config['backend'],
        name='watch-splits', size=2
    )

    if config['virtualenv_cmd']:
        queue.add_header_command(config['virtualenv_cmd'])

    _submit_split_jobs(base_fpath, queue)

    if config['verbose']:
        queue.rprint()

    if config['run']:
        if config['serial']:
            queue.serial_run()
        else:
            queue.run()
        agg_state = queue.monitor()
        try:
            if not config['keep_sessions']:
                if not agg_state['errored']:
                    queue.kill()
        except Exception:
            pass

    return queue

main = prep_splits

if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        python -m watch.cli.prepare_splits \
            --base_fpath=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json \
            --run=0 --backend=serial

        python -m watch.cli.prepare_splits \
            --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/foo.kwcoco.json \
            --run=1 --backend=slurm

        DVC_DPATH=$(python -m watch.cli.find_dvc)
        python -m watch.cli.prepare_splits \
            --base_fpath=$DVC_DPATH/Drop1-Aligned-L1-2022-01/data.kwcoco.json \
            --run=1 --backend=slurm
    """
    main(cmdline=True)

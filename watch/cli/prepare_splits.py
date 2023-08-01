#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
CommandLine:
    xdoctest -m watch.cli.prepare_splits __doc__

Example:
    >>> from watch.cli.prepare_splits import *  # NOQA
    >>> base_fpath = 'data.kwcoco.json'
    >>> config = {
    >>>     'base_fpath': './bundle/data.kwcoco.json',
    >>>     'virtualenv_cmd': 'conda activate watch',
    >>>     'constructive_mode': False,
    >>>     'run': 0,
    >>>     'cache': False,
    >>>     'backend': 'serial',
    >>>     'verbose': 0,
    >>> }
    >>> queue = prep_splits(cmdline=False, **config)
    >>> config['backend'] = 'slurm'
    >>> queue = prep_splits(cmdline=False, **config)
    >>> queue.print_commands()
    >>> config['backend'] = 'tmux'
    >>> queue = prep_splits(cmdline=False, **config)
    >>> queue.print_commands()
    >>> config['backend'] = 'serial'
    >>> queue = prep_splits(cmdline=False, **config)
    >>> queue.print_commands()

"""

import scriptconfig as scfg
import ubelt as ub


class PrepareSplitsConfig(scfg.DataConfig):
    """
    This generates the bash commands necessary to split a base kwcoco file into
    the standard train / validation splits.

    Ignore:
        base_fpath = 'imganns*.kwcoco.*'
    """
    __default__ = {
        'base_fpath': scfg.Value(None, nargs='+', help='input kwcoco files to be joined into splits (or one base coco file to split in non-constructive mode)', position=1),

        'dst_dpath': scfg.Value(None, help='location to write the new kwcoco files. If unspecfied uses the folder of the first input kwcoco file'),

        'virtualenv_cmd': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your bashrc
            does not start it by default.''')),
        'run': scfg.Value(True, help='if True execute the pipeline'),
        'cache': scfg.Value(0, help='if True skip completed results'),

        'backend': scfg.Value('tmux', help='can be serial, tmux, or slurm. Using tmux is recommended.'),
        'with_textual': scfg.Value('auto', help='setting for cmd-queue monitoring'),
        'other_session_handler': scfg.Value('ask', help='for tmux backend only. How to handle conflicting sessions. Can be ask, kill, or ignore, or auto'),

        'constructive_mode': scfg.Value(True, help='if True use the new constructive mode'),

        'verbose': scfg.Value(1, help=''),
        'workers': scfg.Value(2, help='', alias=['tmux_workers']),

        'suffix': scfg.Value('', help='suffix for the output split filenames'),

        'splits': scfg.Value('*', help='restrict to only a specific split')
    }


# TODO: should be some sort of external file we read / define
VALI_REGIONS_SPLITS = {
    'split1': {
        'KR_R001',
        'KR_R002',
    },
    'split2': {
        'BR_R002',
        'NZ_R001',
    },
    'split3': {
        'AE_R001',
        'US_R004',
    },
    'split4': {
        'BR_R001',
        'LT_R001',
        'US_R004',
    },
    'split5': {
        'BR_R001',
        'US_R001',
        'CH_R001',
    },
    'split6': {
        'KR_R002',  # can we do better on KR2 by training on KR1?
    },
}

IGNORE_REGIONS = {
    'CN_C001',
}


def _submit_constructive_split_jobs(base_fpath, dst_dpath, suffix, queue, config, depends=[]):
    """
    new method for splits to construct them from previouly partitioned files
    """
    from kwutil import util_path
    from kwutil import util_pattern
    split_pat = util_pattern.MultiPattern.coerce(config.splits)

    import shlex
    partitioned_fpaths = util_path.coerce_patterned_paths(base_fpath)
    print('partitioned_fpaths = {}'.format(ub.urepr(partitioned_fpaths, nl=1)))

    if dst_dpath is None:
        dst_dpath = ub.Path(partitioned_fpaths[0]).parent  # Hack

    full_fpath = dst_dpath / 'data.kwcoco.zip'

    for split, vali_regions in VALI_REGIONS_SPLITS.items():
        if not split_pat.match(split):
            continue

        train_split_fpath = dst_dpath / f'data_train_{split}.kwcoco.zip'
        vali_split_fpath = dst_dpath / f'data_vali_{split}.kwcoco.zip'
        if suffix:
            train_split_fpath = dst_dpath / f'data_train_{suffix}_{split}.kwcoco.zip'
            vali_split_fpath = dst_dpath / f'data_vali_{suffix}_{split}.kwcoco.zip'
        else:
            train_split_fpath = dst_dpath / f'data_train_{split}.kwcoco.zip'
            vali_split_fpath = dst_dpath / f'data_vali_{split}.kwcoco.zip'
        train_parts = []
        vali_parts = []
        for fpath in partitioned_fpaths:
            if any(v in fpath.name for v in IGNORE_REGIONS):
                ...
            elif any(v in fpath.name for v in vali_regions):
                vali_parts.append(fpath)
            else:
                train_parts.append(fpath)

        train_parts_str = ' '.join([shlex.quote(str(p)) for p in train_parts])
        vali_parts_str = ' '.join([shlex.quote(str(p)) for p in vali_parts])

        command = ub.codeblock(
            fr'''
            python -m kwcoco union \
                --remember_parent=True \
                --src {vali_parts_str} \
                --dst {vali_split_fpath}
            ''')
        queue.submit(command, begin=1, depends=depends, log=False)

        command = ub.codeblock(
            fr'''
            python -m kwcoco union \
                --remember_parent=True \
                --src {train_parts_str} \
                --dst {train_split_fpath}
            ''')
        queue.submit(command, depends=depends, log=False)

    all_parts_str = ' '.join([shlex.quote(str(p)) for p in partitioned_fpaths])
    command = ub.codeblock(
        fr'''
        python -m kwcoco union \
            --remember_parent=True \
            --src {all_parts_str} \
            --dst {full_fpath}
        ''')
    queue.submit(command, depends=depends, log=False)


def _submit_split_jobs(base_fpath, queue, depends=[]):
    """
    Populates a Serial, Slurm, or Tmux Queue with jobs
    """

    base_fpath = ub.Path(base_fpath)

    splits = {
        # 'nowv': base_fpath.augment(stemsuffix='_nowv', multidot=True),

        'train_split1': base_fpath.augment(stemsuffix='_train_split1', multidot=True),
        'train_split2': base_fpath.augment(stemsuffix='_train_split2', multidot=True),
        'train_split3': base_fpath.augment(stemsuffix='_train_split3', multidot=True),
        'train_split4': base_fpath.augment(stemsuffix='_train_split4', multidot=True),
        'train_split5': base_fpath.augment(stemsuffix='_train_split5', multidot=True),
        'train_split6': base_fpath.augment(stemsuffix='_train_split6', multidot=True),
        # 'nowv_train': base_fpath.augment(stemsuffix='_nowv_train', multidot=True),
        # 'wv_train': base_fpath.augment(stemsuffix='_wv_train', multidot=True),
        # 's2_wv_train': base_fpath.augment(stemsuffix='_s2_wv_train', multidot=True),

        'vali_split1': base_fpath.augment(stemsuffix='_vali_split1', multidot=True),
        'vali_split2': base_fpath.augment(stemsuffix='_vali_split2', multidot=True),
        'vali_split3': base_fpath.augment(stemsuffix='_vali_split3', multidot=True),
        'vali_split4': base_fpath.augment(stemsuffix='_vali_split4', multidot=True),
        'vali_split5': base_fpath.augment(stemsuffix='_vali_split5', multidot=True),
        'vali_split6': base_fpath.augment(stemsuffix='_vali_split6', multidot=True),
        # 'nowv_vali': base_fpath.augment(stemsuffix='_nowv_vali', multidot=True),
        # 'wv_vali': base_fpath.augment(stemsuffix='_wv_vali', multidot=True),
        # 's2_wv_vali': base_fpath.augment(stemsuffix='_s2_wv_vali', multidot=True),
    }

    # train_region_selector = '(' + ' or '.join(['(.name ==  "{}")'.format(n) for n in (ignore_regions | vali_regions)]) + ') | not'
    # vali_region_selector = ' or '.join(['(.name ==  "{}")'.format(n) for n in (vali_regions)])

    for split, vali_regions in VALI_REGIONS_SPLITS.items():

        train_key = f'train_{split}'
        vali_key = f'vali_{split}'

        train_region_selector = '(' + ' or '.join(['(.name | startswith("{}"))'.format(n) for n in (IGNORE_REGIONS | vali_regions)]) + ') | not'
        vali_region_selector = ' or '.join(['(.name | startswith("{}"))'.format(n) for n in (vali_regions)])

        split_jobs = {}
        # Perform train/validation splits with and without worldview
        command = ub.codeblock(
            fr'''
            python -m kwcoco subset \
                --src {base_fpath} \
                --dst {splits[train_key]} \
                --select_videos '{train_region_selector}'
            ''')
        split_jobs['train'] = queue.submit(command, begin=1, depends=depends, log=False)

        # Perform vali/validation splits with and without worldview
        command = ub.codeblock(
            fr'''
            python -m kwcoco subset \
                --src {base_fpath} \
                --dst {splits[vali_key]} \
                --select_videos '{vali_region_selector}'
            ''')
        split_jobs['vali'] = queue.submit(command, depends=depends, log=False)
    return queue


def prep_splits(cmdline=False, **kwargs):
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.

    TODO:
        - [ ] Option to just dump the serial bash script that does everything.
    """
    config = PrepareSplitsConfig.cli(data=kwargs, cmdline=cmdline, strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1)))

    if config['base_fpath'] == 'auto':
        # Auto hack.
        raise NotImplementedError
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
        # base_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
    else:
        base_fpath = config['base_fpath']

    import cmd_queue
    queue = cmd_queue.Queue.create(
        backend=config['backend'],
        name='watch-splits', size=config['workers'],
    )

    if config['virtualenv_cmd']:
        queue.add_header_command(config['virtualenv_cmd'])

    if config.dst_dpath is not None:
        dst_dpath = ub.Path(config.dst_dpath)
    else:
        dst_dpath = None
    suffix = config.suffix

    if config['constructive_mode']:
        _submit_constructive_split_jobs(base_fpath, dst_dpath, suffix, queue, config)
    else:
        raise NotImplementedError('non-constructive mode is no longer supported')
        print('WARNING: non-constructive mode has not been maintained')
        _submit_split_jobs(base_fpath, queue)

    if config['verbose']:
        queue.rprint()

    if config['run']:
        queue.run(block=True, with_textual=config['with_textual'])

    return queue


main = prep_splits

if __name__ == '__main__':
    """
    CommandLine:
        DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data)
        BASE_FPATH=$DVC_DATA_DPATH/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data.kwcoco.json
        python -m watch.cli.prepare_splits \
            --base_fpath=$BASE_FPATH \
            --backend=serial --run=0
    """
    main(cmdline=True)

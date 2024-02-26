#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
TODO:
    move to queue_cli

CommandLine:
    xdoctest -m geowatch.cli.prepare_splits __doc__

Example:
    >>> from geowatch.cli.prepare_splits import *  # NOQA
    >>> dpath = ub.Path.appdir('geowatch', 'tests', 'prep_splits').ensuredir()
    >>> (dpath / 'KR_R001.kwcoco.zip').touch()
    >>> (dpath / 'KR_R002.kwcoco.zip').touch()
    >>> (dpath / 'BR_R002.kwcoco.zip').touch()
    >>> config = {
    >>>     'base_fpath': dpath / '*.kwcoco.zip',
    >>>     'virtualenv_cmd': 'conda activate geowatch',
    >>>     'constructive_mode': True,
    >>>     'run': 0,
    >>>     'cache': False,
    >>>     'backend': 'serial',
    >>>     'splits': 'split6',
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


CommandLine:

    DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
    python -m geowatch.cli.prepare_splits \
        --src_kwcocos="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-NoMask/*/imganns-*.kwcoco.zip \
        --dst_dpath "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-NoMask \
        --suffix=rawbands \
        --backend=tmux --tmux_workers=6 \
        --splits=split6 \
        --run=0
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
    src_kwcocos = scfg.Value(None, alias=['base_fpath'], position=1, help=ub.paragraph(
            '''
            input kwcoco files to be joined into splits
            '''), nargs='+')
    dst_dpath = scfg.Value(None, help=ub.paragraph(
            '''
            location to write the new kwcoco files. If unspecfied uses
            the folder of the first input kwcoco file
            '''))
    suffix = scfg.Value('', help=ub.paragraph(
            '''
            destination suffix for the output split filenames
            '''))
    virtualenv_cmd = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your
            bashrc does not start it by default.
            '''))
    run = scfg.Value(True, help='if True execute the pipeline')
    cache = scfg.Value(0, help='if True skip completed results')
    backend = scfg.Value('tmux', help=ub.paragraph(
            '''
            can be serial, tmux, or slurm. Using tmux is recommended.
            '''))
    with_textual = scfg.Value('auto', help='setting for cmd-queue monitoring')
    other_session_handler = scfg.Value('ask', help=ub.paragraph(
            '''
            for tmux backend only. How to handle conflicting sessions.
            Can be ask, kill, or ignore, or auto
            '''))
    constructive_mode = scfg.Value(True, help='if True use the new constructive mode')
    verbose = scfg.Value(1, help='')
    workers = scfg.Value(2, alias=['tmux_workers'], help='')
    splits = scfg.Value('*', help='restrict to only a specific split')

    add_detail_suffix = scfg.Value(True, help=ub.paragraph(
        '''
        if True, add a suffix info to the output paths to disambiguate splits
        with different inputs
        '''))


imerit_vali_regions = {'CN_C000', 'KW_C001', 'SA_C001', 'CO_C001', 'VN_C002'}

# TODO: should be some sort of external file we read / define
VALI_REGIONS_SPLITS = {
    'split1': {
        'KR_R001',
        'KR_R002',
    } | imerit_vali_regions,
    'split2': {
        'BR_R002',
        'NZ_R001',
    } | imerit_vali_regions,
    'split3': {
        'AE_R001',
        'US_R004',
    } | imerit_vali_regions,
    'split4': {
        'BR_R001',
        'LT_R001',
        'US_R004',
    } | imerit_vali_regions,
    'split5': {
        'BR_R001',
        'US_R001',
        'CH_R001',
    } | imerit_vali_regions,
    'split6': {
        'KR_R002',  # can we do better on KR2 by training on KR1?
    } | imerit_vali_regions,
}

IGNORE_REGIONS = {
    # 'CN_C001',
}


def _submit_constructive_split_jobs(base_fpath, dst_dpath, suffix, queue, config, depends=[]):
    """
    new method for splits to construct them from previouly partitioned files
    """
    from kwutil import util_path
    from kwutil import util_pattern
    split_pat = util_pattern.MultiPattern.coerce(config.splits)

    import shlex
    partitioned_fpaths = sorted(util_path.coerce_patterned_paths(base_fpath))
    print('partitioned_fpaths = {}'.format(ub.urepr(partitioned_fpaths, nl=1)))

    if dst_dpath is None:
        dst_dpath = ub.Path(partitioned_fpaths[0]).parent  # Hack

    full_fpath = dst_dpath / 'data.kwcoco.zip'

    if config.add_detail_suffix:
        path_to_hash = {}
        for p in ub.ProgIter(partitioned_fpaths, desc='compute hashes'):
            path_to_hash[p] = ub.hash_file(p)

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

        if config.add_detail_suffix:
            train_hashid = ub.hash_data(sorted([path_to_hash[p] for p in train_parts]))[0:8]
            vali_hashid = ub.hash_data(sorted([path_to_hash[p] for p in vali_parts]))[0:8]

            train_detail_suffix = f'_n{len(train_parts):03d}_{train_hashid}'
            vali_detail_suffix = f'_n{len(vali_parts):03d}_{vali_hashid}'

            train_hashed_fpath = train_split_fpath.augment(stemsuffix=train_detail_suffix, multidot=1)
            vali_hashed_fpath = vali_split_fpath.augment(stemsuffix=vali_detail_suffix, multidot=1)
        else:
            train_hashed_fpath = train_split_fpath
            vali_hashed_fpath = vali_split_fpath

        train_parts_str = ' '.join([shlex.quote(str(p)) for p in train_parts])
        vali_parts_str = ' '.join([shlex.quote(str(p)) for p in vali_parts])

        if len(vali_parts):
            command = ub.codeblock(
                fr'''
                python -m kwcoco union \
                    --remember_parent=True \
                    --src {vali_parts_str} \
                    --dst {vali_hashed_fpath}
                ''')
            vali_job = queue.submit(command, begin=1, depends=depends, log=False)
            if config.add_detail_suffix:
                # Symlink to original locations
                vali_job = queue.submit(f'ln -sf {vali_hashed_fpath} {vali_split_fpath}', begin=1, depends=vali_job, log=False)

        if len(train_parts):
            command = ub.codeblock(
                fr'''
                python -m kwcoco union \
                    --remember_parent=True \
                    --src {train_parts_str} \
                    --dst {train_hashed_fpath}
                ''')
            train_job = queue.submit(command, depends=depends, log=False)
            if config.add_detail_suffix:
                # Symlink to original locations
                train_job = queue.submit(f'ln -sf {train_hashed_fpath} {train_split_fpath}', begin=1, depends=train_job, log=False)

    if 0:
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
        import geowatch
        dvc_dpath = geowatch.find_dvc_dpath()
        # base_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
    else:
        base_fpath = config['base_fpath']

    import cmd_queue
    queue = cmd_queue.Queue.create(
        backend=config['backend'],
        name='geowatch-splits', size=config['workers'],
    )

    if config['virtualenv_cmd']:
        queue.add_header_command(config['virtualenv_cmd'])

    if config.dst_dpath is not None:
        dst_dpath = ub.Path(config.dst_dpath)
    else:
        dst_dpath = None
    suffix = config.suffix

    if not config['constructive_mode']:
        raise NotImplementedError('non-constructive mode is no longer supported')
        print('WARNING: non-constructive mode has not been maintained')
        _submit_split_jobs(base_fpath, queue)

    _submit_constructive_split_jobs(base_fpath, dst_dpath, suffix, queue, config)

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
        python -m geowatch.cli.prepare_splits \
            --base_fpath=$BASE_FPATH \
            --backend=serial --run=0
    """
    main(cmdline=True)

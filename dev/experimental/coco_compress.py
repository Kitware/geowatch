import scriptconfig as scfg
import ubelt as ub
import kwcoco


class CocoCompressConfig(scfg.DataConfig):
    """
    Performs different operations that can reduce the kwcoco file size

    TODO: separate into a general fixup step for kwcoco proper and one for
    SMART WATCH.
    """
    src = scfg.Value(None, nargs='+', help='one or more input kwcoco paths', position=1)
    dst = scfg.Value(None, help='output kwcoco dataset path')

    fix_duplicate_infos = scfg.Value(True, help='Finds and removes duplicate info items')
    fix_resolution = scfg.Value(True, help='Ensures target_gsd is matched by resolution')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch/dev/experimental'))
        >>> from coco_compress import *  # NOQA
        >>> cmdline = 0
        >>> src_fpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testsc/pred/flat/sc_poly/sc_poly_id_fbe0aaed/poly.kwcoco.json').expand()
        >>> dst_fpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testsc/pred/flat/sc_poly/sc_poly_id_fbe0aaed/poly.kwcoco.json').expand()
        >>> kwargs = dict(
        >>>     src=src_fpath,
        >>>     dst=dst_fpath,
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = CocoCompressConfig.legacy(cmdline=cmdline, data=kwargs)
    print('config = ' + ub.urepr(dict(config), nl=1))

    if config.dst is None:
        raise ValueError('needs an output file')

    dset = kwcoco.CocoDataset.coerce(config.src)

    if config.fix_duplicate_infos:
        import json
        seen_ = set()
        fixed_infos = []
        for info in ub.ProgIter(dset.dataset['info'], desc='deduplicate info'):
            key = json.dumps(info)
            if key not in seen_:
                fixed_infos.append(info)
                seen_.add(key)
        print(f'New info size: {len(fixed_infos)}')
        dset.dataset['info'] = fixed_infos

    if config.fix_resolution:
        for video in dset.dataset['videos']:
            if 'target_gsd' in video and 'resolution' not in video:
                video['resolution'] = '{} GSD'.format(video['target_gsd'])

    dset.fpath = config.dst
    dset.dump()


def schedule_problem_fixes():
    # dpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testsc/pred/flat/sc_poly').expand()
    # sub_dpaths = dpath.ls()

    # dpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testsc').expand()
    dpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/').expand()

    possible_kwcoco_fpaths = list(dpath.glob('*/*/flat/*/*/*.kwcoco.json'))
    candidate_rows = []
    for src_fpath in possible_kwcoco_fpaths:
        num_megabytes = src_fpath.stat().st_size // (2 ** 20)
        candidate_rows.append(
            {'fpath': src_fpath, 'size_mb': num_megabytes}
        )

    import pandas as pd
    df = pd.DataFrame(candidate_rows)
    total_mb = df['size_mb'].sum()
    print(f'total_mb={total_mb}')
    df = df.sort_values('size_mb')
    shrink_candidates = df[df['size_mb'] > 300]

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=2)
    for fpath in shrink_candidates['fpath']:
        queue.submit(f'python ~/code/watch/dev/experimental/coco_compress.py --src {fpath} --dst {fpath}')

    queue.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/experimental/coco_compress.py
    """
    main()

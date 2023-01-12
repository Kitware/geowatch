import scriptconfig as scfg
import ubelt as ub
import kwcoco


class CocoCompressConfig(scfg.DataConfig):
    """
    Performs different operations that can reduce the kwcoco file size

    TODO: separate into a general fixup step for kwcoco proper and one for
    SMART WATCH.
    """
    src = scfg.Value(None, help='input kwcoco path', position=1)
    dst = scfg.Value(None, help='output kwcoco dataset path', position=2)

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

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/dev/experimental'))
        from coco_compress import *  # NOQA
        cmdline = 0
        src_fpath = ub.Path('data_vali_split1.kwcoco.json').expand()
        kwargs = dict(
            src=src_fpath,
            dst=src_fpath,
        )

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
        num_fixed = 0
        for video in dset.dataset['videos']:
            if 'target_gsd' in video and 'resolution' not in video:
                num_fixed += 1
                video['resolution'] = '{} GSD'.format(video['target_gsd'])
        print(f'Converted target_gsd -> resolution: {num_fixed}')

    if 0:
        from kwcoco.coco_image import CocoImage
        num_removed_deprecated_fields = 0
        num_missing_base_geos_corners = 0

        for img in dset.dataset['images']:
            coco_img = CocoImage(img)
            deprecated_fields = {
                'wgs84_corners',
                'wgs84_crs_info',
                'utm_corners',
                'utm_crs_info',
            }
            assert 'geos_corners' in coco_img.img
            if 'geos_corners' not in img:
                primary = coco_img.primary_asset()
                assert False, 'should not happen?'
                if 'geos_corners' in primary:
                    num_missing_base_geos_corners += 1
                    # img['geos_corners'] = primary['geos_corners']

            for obj in ub.flatten([coco_img.iter_asset_objs(), [coco_img.img]]):
                for key in deprecated_fields:
                    if key in obj:
                        num_removed_deprecated_fields += 1
                        obj.pop(key)

        print(f'num_missing_base_geos_corners={num_missing_base_geos_corners}')
        print(f'num_removed_deprecated_fields={num_removed_deprecated_fields}')

    dset.fpath = config.dst
    dset.dump()


def schedule_problem_fixes():
    # dpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testsc/pred/flat/sc_poly').expand()
    # sub_dpaths = dpath.ls()

    # dpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testsc').expand()
    # dpath = ub.Path('~/remote/toothbrush/data/dvc-repos/smart_expt_dvc/').expand()
    # possible_kwcoco_fpaths = list(dpath.glob('*/*/flat/*/*/*.kwcoco.json'))
    dpath = ub.Path('.')
    possible_kwcoco_fpaths = list(dpath.glob('*.kwcoco.json'))

    candidate_rows = []
    for src_fpath in possible_kwcoco_fpaths:
        num_megabytes = src_fpath.stat().st_size // (2 ** 20)
        candidate_rows.append(
            {'fpath': src_fpath, 'size_mb': num_megabytes}
        )

    import pandas as pd
    df = pd.DataFrame(candidate_rows)
    df = df.sort_values('size_mb')

    import rich
    rich.print(df.to_string())

    total_mb = df['size_mb'].sum()
    print(f'total_mb={total_mb}')
    shrink_candidates = df
    # shrink_candidates = df[df['size_mb'] > 300]

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

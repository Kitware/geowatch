"""
Combine kwcoco files with different "auxiliary" features into a single kwcoco
file.
"""
import ubelt as ub
import scriptconfig as scfg


class CocoCombineFeatures(scfg.DataConfig):
    """
    Combine kwcoco files with different "auxiliary" features into a single
    kwcoco file.

    TODO:
        - [ ] This might go in kwcoco proper? This could be folded into "union"
    """
    __default__ = {
        'src': scfg.Value([], nargs='+', help='path to datasets. The first one will be the "base"', position=1),

        'dst': scfg.Value(None, help='dataset to write to'),

        'io_workers': scfg.Value('avail', help='number of workers used to read multiple datasets'),

        'absolute': scfg.Value(False, isflag=True, help='if True, use absolute paths'),
    }


def combine_auxiliary_features(dst_dset, src_dsets):
    """
    Example:
        >>> from watch.cli.coco_combine_features import *  # NOQA
        >>> import kwcoco
        >>> base = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> dset1 = base.copy()
        >>> dset2 = base.copy()
        >>> dset3 = base.copy()
        >>> dset4 = base.copy()
        >>> for img in dset1.index.imgs.values():
        >>>     del img['auxiliary'][0::3]
        >>> for img in dset2.index.imgs.values():
        >>>     del img['auxiliary'][1::3]
        >>> dset2.remove_images([2, 3])
        >>> for img in dset3.index.imgs.values():
        >>>     del img['auxiliary'][2::3]
        >>> dset3.remove_images([2, 3])
        >>> for img in dset4.index.imgs.values():
        >>>     del img['auxiliary'][0::2]
        >>> dset4.remove_images([2, 3])
        >>> dst_dset = dset1
        >>> src_dsets = [dset2, dset3, dset4]
        >>> for img in dset1.index.imgs.values():
        ...     assert len(img['auxiliary']) != 5
        >>> dst_dset = combine_auxiliary_features(dst_dset, src_dsets)
        >>> lens1 = list(map(len, dset1.images(set(dset1.imgs) - {2, 3}).lookup('auxiliary')))
        >>> assert ub.allsame([5] + lens1)
        >>> lens2 = list(map(len, dset1.images({2, 3}).lookup('auxiliary')))
        >>> assert ub.allsame([3] + lens2)
    """

    for src_dset in src_dsets:
        gids1, gids2, report = associate_images(dst_dset, src_dset)
        print('report = {!r}'.format(report))
        for gid1, gid2 in zip(gids1, gids2):
            dst_img = dst_dset.index.imgs[gid1]
            src_img = src_dset.index.imgs[gid2]
            dst_auxiliary = dst_img.get('auxiliary')
            src_auxiliary = src_img.get('auxiliary')
            if src_auxiliary is None:
                src_auxiliary = []  # nothing will happen in this case
            if dst_auxiliary is None:
                dst_auxiliary = dst_img['auxiliary'] = []
            have_channels = set(aux.get('channels') for aux in dst_auxiliary)
            assert src_img['name'] == dst_img['name']
            for src_aux in src_auxiliary:
                if src_aux['channels'] not in have_channels:
                    have_channels.add(src_aux['channels'])
                    dst_auxiliary.append(src_aux)
    return dst_dset


def main(cmdline=True, **kwargs):
    """
    Example:
        >>> from watch.cli import coco_combine_features
        >>> import watch
        >>> dset = watch.coerce_kwcoco('watch-msi')
        >>> dpath = ub.Path.appdir('watch/tests/combine_fetures').ensuredir()
        >>> # Breakup the data into two parts with different features
        >>> dset1 = dset.copy()
        >>> dset2 = dset.copy()
        >>> dset1.fpath = dpath / 'part1.kwcoco.json'
        >>> dset2.fpath = dpath / 'part2.kwcoco.json'
        >>> # Remove all but the first asset from dset1
        >>> for coco_img in dset1.images().coco_images:
        ...     del coco_img.img['auxiliary'][1:]
        >>> # Remove the first asset from dset2
        >>> for coco_img in dset2.images().coco_images:
        ...     del coco_img.img['auxiliary'][0]
        >>> dset1.dump()
        >>> dset2.dump()
        >>> from watch.utils import kwcoco_extensions
        >>> chan_stats0 = kwcoco_extensions.coco_channel_stats(dset)['chan_hist']
        >>> chan_stats1 = kwcoco_extensions.coco_channel_stats(dset1)['chan_hist']
        >>> chan_stats2 = kwcoco_extensions.coco_channel_stats(dset2)['chan_hist']
        >>> assert chan_stats1 != chan_stats0, 'channels should be different'
        >>> dst_fpath = dpath / 'combo.kwcoco.json'
        >>> kwargs = {
        >>>     'src': [str(dset1.fpath), str(dset2.fpath)],
        >>>     'dst': str(dst_fpath),
        >>> }
        >>> cmdline = 0
        >>> coco_combine_features.main(cmdline=cmdline, **kwargs)
        >>> dst_dset = watch.coerce_kwcoco(dst_fpath)
        >>> chan_stats3 = kwcoco_extensions.coco_channel_stats(dst_dset)['chan_hist']
        >>> assert chan_stats3 == chan_stats0, (
        >>>     'combine features should have the same as the original dset')

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # xdoctest: +SKIP
        >>> # drop1-S2-L8-aligned-old deprecated
        >>> from watch.cli.coco_combine_features import *  # NOQA
        >>> import os
        >>> from watch.utils.util_path import coercepath
        >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
        >>> dvc_dpath = coercepath(os.environ.get('DVC_DPATH', _default))
        >>> fpath1 = dvc_dpath / 'drop1-S2-L8-aligned/data.kwcoco.json'
        >>> #fpath1 = dvc_dpath / 'drop1-S2-L8-aligned-old/data.kwcoco.json'
        >>> fpath2 = dvc_dpath / 'drop1-S2-L8-aligned-old/uky_invariants.kwcoco.json'
        >>> fpath3 = dvc_dpath / 'drop1-S2-L8-aligned/_testcombo.kwcoco.json'
        >>> assert fpath1.exists()
        >>> assert fpath2.exists()
        >>> cmdline = False
        >>> kwargs = {
        >>>     'src': [str(fpath1), str(fpath2)],
        >>>     'dst': str(fpath3),
        >>> }
        >>> main(cmdline, **kwargs)
    """
    import kwcoco
    config = CocoCombineFeatures.cli(data=kwargs, cmdline=cmdline)
    import rich
    rich.print(ub.urepr(config))

    fpaths = config['src']

    dset_iter = kwcoco.CocoDataset.coerce_multiple(
        fpaths, workers=config.io_workers)
    dset_list = []
    for dset in dset_iter:
        if config['absolute']:
            dset.reroot(absolute=True)
        dset_list.append(dset)

    dset1 = dset_list[0]

    for dset2 in dset_list[1:]:
        gids1, gids2, report = associate_images(dset1, dset2)

    src_dsets = dset_list[1:]
    dst_dset = dset_list[0]
    dst_dset.fpath = config['dst']

    dst_dset = combine_auxiliary_features(dst_dset, src_dsets)

    missing_hist = ub.ddict(lambda: 0)
    channel_specs = []
    # Check which images have which features (did we miss any?)
    for _gid, dst_img in ub.ProgIter(dst_dset.index.imgs.items(),
                                     total=dst_dset.n_images,
                                     desc='checking features'):
        img_channels = set()
        for aux in dst_img.get('auxiliary'):
            img_channels.add(aux['channels'])
        channel_specs.append(img_channels)
    all_channels = set.union(*channel_specs)
    for spec in channel_specs:
        missing = all_channels - spec
        if missing:
            for k in missing:
                missing_hist[k] += 1

    if missing_hist:
        print('missing_hist = {!r}'.format(missing_hist))

    print('dump dst_dset.fpath = {!r}'.format(dst_dset.fpath))
    dst_dset.dump(dst_dset.fpath, newlines=True)


def associate_images(dset1, dset2):
    """
    Hueristic for getting pairs of images that correspond between two datasets
    """
    dset1_img_names = set(dset1.index.name_to_img)
    dset2_img_names = set(dset2.index.name_to_img)
    common_names = dset1_img_names & dset2_img_names
    dset1_missing_img_names = dset1_img_names - common_names
    dset2_missing_img_names = dset2_img_names - common_names
    report = {}
    report.update({
        'num_name_common': len(common_names),
        'num_name_missing1': len(dset1_missing_img_names),
        'num_name_missing2': len(dset2_missing_img_names),
    })

    gids1 = []
    gids2 = []
    for name in common_names:
        img1 = dset1.index.name_to_img[name]
        img2 = dset2.index.name_to_img[name]
        gids1.append(img1['id'])
        gids2.append(img2['id'])

    return gids1, gids2, report


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/coco_combine_features.py
        python -m watch.cli.coco_combine_features
    """
    main(cmdline=True)

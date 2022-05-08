"""
Combine kwcoco files with different "auxiliary" features into a single kwcoco
file.
"""


import kwcoco
import ubelt as ub
import scriptconfig as scfg


class CocoCombineFeatures(scfg.Config):
    """
    Combine kwcoco files with different "auxiliary" features into a single
    kwcoco file.

    TODO:
        - [ ] This might go in kwcoco proper? This could be folded into "union"
    """
    default = {
        'src': scfg.Value([], nargs='+', help='path to datasets. The first one will be the "base"', position=1),

        'dst': scfg.Value(None, help='dataset to write to'),

        'absolute': scfg.Value(False, help='if True, use absolute paths'),
    }


def combine_auxiliary_features(dst_dset, src_dsets):
    """
    Example:
        >>> from watch.cli.coco_combine_features import *  # NOQA
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
    config = CocoCombineFeatures(default=kwargs, cmdline=cmdline)

    fpaths = config['src']

    dset_list = []
    for fpath in ub.ProgIter(fpaths, desc='read src datasets'):
        dset = kwcoco.CocoDataset.coerce(fpath)
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

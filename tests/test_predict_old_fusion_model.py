import ubelt as ub


def request_dvc_path(fpath):
    """
    If the file is in a DVC directory and the user has permissions,
    attempt to pull the file from a remote if it doesn't exist locally.
    """
    if not fpath.exists():
        dvc_fpath = fpath.augment(stem=fpath.name, ext='.dvc')
        if dvc_fpath.exists():
            remote = 'aws'  # parametarize
            ub.cmd(f'dvc pull -r {remote} {dvc_fpath.name}',
                   cwd=dvc_fpath.parent, verbose=2, check=True)
        raise Exception('File {} not exist in a DVC directory'.format(dvc_fpath))


def test_predict_old_fusion_model():
    """
    Check that the old fusion models still work for prediction.

    This tests requires program data as is not exepcted to run on CI in this
    state. We should have a test for our latest-and-greatest models though.
    """
    import watch
    import kwcoco
    import pytest
    from watch.utils import kwcoco_extensions
    import ubelt as ub

    try:
        dvc_dpath = watch.find_smart_dvc_dpath()
    except Exception:
        pytest.skip('dvc path does not exist')

    # model_fpath = dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_raw_v39/SC_smt_it_stm_p8_newanns_weighted_raw_v39_epoch=53-step=2311901.pt'

    # model_fpath = dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_st_s12_newanns_weighted_rgb_v22/SC_smt_it_st_s12_newanns_weighted_rgb_v22_epoch=117-step=5051933.pt'
    model_fpath = dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_raw_v39/SC_smt_it_stm_p8_newanns_weighted_raw_v39_epoch=53-step=2311901.pt'

    request_dvc_path(model_fpath)

    # from watch.tasks.fusion import utils
    # method = utils.load_model_from_package(model_fpath)

    coco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
    if not model_fpath.exists():
        pytest.skip('expected model does not exist')

    if not coco_fpath.exists():
        pytest.skip('expected data does not exist')

    dset = kwcoco.CocoDataset(coco_fpath)

    vidid = dset.videos().peek()['id']
    gids = list(dset.images(vidid=vidid))[0:3]
    subset = dset.subset(gids)

    output_dpath = ub.Path(ub.ensure_app_cache_dir('watch/tests/pred/oldmodel'))
    # output_dpath.delete().ensuredir()

    subset.reroot(absolute=True)
    subset.fpath = str(output_dpath / 'test_input.kwcoco.json')

    walker = ub.IndexableWalker(subset.dataset['images'])
    tofix = []
    for path, value in walker:
        # Hack for my sanity
        if path[-1] == 'shear':
            transform_dict = walker[path[:-1]]
            tofix.append(transform_dict)
    import kwimage
    for transform_dict in tofix:
        fixed = kwimage.Affine.coerce(transform_dict).concise()
        transform_dict.clear()
        transform_dict.update(fixed)

    stats = kwcoco_extensions.coco_channel_stats(subset)
    print('stats = {}'.format(ub.repr2(stats, nl=2)))
    subset.dump(subset.fpath, newlines=True)

    pred_fpath = output_dpath / 'pred_bundle/pred.kwcoco.json'
    from watch.tasks.fusion import predict
    pred_kwargs = {
        'test_dataset': subset.fpath,
        'pred_dataset': pred_fpath,
        'package_fpath': model_fpath,
        'chip_overlap': 0.0,
        'gpus': 0,
    }
    kwargs = pred_kwargs  # NOQA
    predict.predict(**pred_kwargs)

    """
    # For devs
    cd $HOME/.cache/watch/tests/pred/oldmodel/
    smartwatch visualize ./test_input.kwcoco.json --viz_dpath=./_viz_test_input
    smartwatch visualize ./pred_bundle/pred.kwcoco.json --viz_dpath=./old_viz_check

    smartwatch intensity_histograms ./pred_bundle/pred.kwcoco.json --dst=./intensity_histo.png --stat probability \
            --valid_range=0:1 --exclude_channels=cirrus

    smartwatch visualize /home/joncrall/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-L1-2022-01/data.kwcoco.json --viz_dpath=./orig_viz_check
    """


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/tests/test_predict_old_fusion_model.py
    """
    test_predict_old_fusion_model()
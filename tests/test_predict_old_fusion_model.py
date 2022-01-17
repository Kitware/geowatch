def test_predict_old_fusion_model():
    """
    Check that the old fusion models still work for prediction
    """
    import watch
    import kwcoco
    import pytest
    from watch.utils import kwcoco_extensions
    import ubelt as ub
    dvc_dpath = watch.find_smart_dvc_dpath()
    # model_fpath = dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_raw_v39/SC_smt_it_stm_p8_newanns_weighted_raw_v39_epoch=53-step=2311901.pt'
    model_fpath = dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v34/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v34_epoch=13-step=599381.pt'

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
    output_dpath.delete().ensuredir()

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
        'gpus': 1,
    }
    kwargs = pred_kwargs  # NOQA
    predict.predict(**pred_kwargs)

    """
    # For devs
    cd /home/joncrall/.cache/watch/tests/pred/oldmodel/
    smartwatch visualize ./test_input.kwcoco.json --viz_dpath=./_viz_test_input
    smartwatch visualize ./pred_bundle/pred.kwcoco.json --viz_dpath=./old_viz_check
    smartwatch visualize /home/joncrall/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-L1-2022-01/data.kwcoco.json --viz_dpath=./orig_viz_check
    """

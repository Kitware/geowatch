
def mask_to_scored_polygons(probs, thresh):
    """
    Example:
        >>> from watch.tasks.fusion.postprocess import *  # NOQA
        >>> import kwimage
        >>> probs = kwimage.Heatmap.random(dims=(64, 64), rng=0).data['class_probs'][0]
        >>> thresh = 0.5
        >>> poly1, score1 = list(mask_to_scored_polygons(probs, thresh))[0]
        >>> # xdoctest: +IGNORE_WANT
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(probs > 0.5)
    """
    import kwimage
    import numpy as np
    # Threshold scores
    hard_mask = probs > thresh
    # Convert to polygons
    polygons = kwimage.Mask(hard_mask, 'c_mask').to_multi_polygon()
    for poly in polygons:
        # Compute a score for the polygon
        # First compute the valid bounds of the polygon
        # And create a mask for only the valid region of the polygon
        box = poly.bounding_box().quantize().to_xywh()
        # Ensure w/h are positive
        box.data[:, 2:4] = np.maximum(box.data[:, 2:4], 1)
        x, y, w, h = box.data[0]
        rel_poly = poly.translate((-x, -y))
        rel_mask = rel_poly.to_mask((h, w)).data
        # Slice out the corresponding region of probabilities
        rel_probs = probs[y:y + h, x:x + w]
        total = rel_mask.sum()
        score = 0 if total == 0 else (rel_mask * rel_probs).sum() / total
        yield poly, score


def accumulate_temporal_predictions_simple_v1(result_dataset):
    """
    find $HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/ -iname "package_*.pt"

    TEST_DATASET=$HOME/remote/yardrat/smart_watch_dvc/drop1-S2-L8-aligned/vali_data.kwcoco.json

    PACKAGE_FPATH=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/checkpoints/epoch=1-step=1431.ckpt

    PACKAGE_FPATH=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/checkpoints/epoch=0-step=715-v1.ckpt



    DVC_DPATH=$HOME/remote/yardrat/smart_watch_dvc
    TEST_DATASET=$DVC_DPATH/drop1-S2-L8-aligned/vali_data.kwcoco.json
    PACKAGE_FPATH=$DVC_DPATH/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/checkpoints/epoch=2-step=2147.ckpt

    SUGGESTIONS="$(python -m watch.tasks.fusion.organize suggest_paths \
        --package_fpath=$PACKAGE_FPATH \
        --test_dataset=$TEST_DATASET)"

    PRED_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
    EVAL_DATASET="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"
    echo $PRED_DATASET
    echo $EVAL_DATASET

    python -m watch.tasks.fusion.predict \
        --test_dataset="$TEST_DATASET" \
        --package_fpath="$PACKAGE_FPATH" \
        --pred_dataset="$PRED_DATASET" \
        --write_probs=True \
        --write_preds=False --gpus=0

    python -m watch.tasks.fusion.evaluate \
        --test_dataset=$TEST_DATASET \
        --pred_dataset=$PRED_DATASET


        --package_fpath=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/packages/package_epoch15_step11360.pt \
        --pred_dataset=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/packages/package_epoch15_step11360/preds/kr_vali/pred.kwcoco.json \
        --write_probs=True \
        --write_preds=False

    Ignore:
        from watch.tasks.fusion.predict import *  # NOQA
        import kwcoco
        result_dataset = kwcoco.CocoDataset.coerce(ub.expandpath(
            '$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/'
            'Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/'
            'lightning_logs/version_0/packages/package_epoch15_step11360/preds/kr_vali/pred.kwcoco.json'))

        pred_fpath = '/home/local/KHQ/jon.crall/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/checkpoints/pred_epoch=0-step=715-v1/vali_data.kwcoco/pred.kwcoco.json'
        result_dataset = kwcoco.CocoDataset.coerce(ub.expandpath(pred_fpath))

        import kwplot
        kwplot.autompl()
        kwplot.imshow(probs)

    Drop1RawLeftRight/runs/DirectCD_smt_it_joint_p8_raw7common_v4/lightning_logs/version_2/checkpoints/epoch=2-step=1241-v3.ckpt
    """
    from watch.utils import kwcoco_extensions
    from watch.utils import util_kwimage
    key = 'change_prob'
    dset = result_dataset

    import kwarray
    import ubelt as ub

    for vidid, video in dset.index.videos.items():
        print('video', ub.dict_isect(video, ['width', 'height']))
        running = kwarray.RunningStats()
        gids = dset.index.vidid_to_gids[vidid]
        for gid in gids:
            img = dset.index.imgs[gid]
            coco_img = kwcoco_extensions.CocoImage(img, dset)
            if key in coco_img.channels:
                img_probs = coco_img.delay(key, space='video').finalize()
                running.update(img_probs)

        probs = running.summarize(axis=2, keepdims=False)['mean']

        thresh = 0.15
        hard_probs = util_kwimage.morphology(probs > thresh, 'close', 3)
        modulated_probs = probs * hard_probs

        scored_polys = list(mask_to_scored_polygons(modulated_probs, thresh))

        # Add each polygon to every images as a track
        import kwimage
        change_cid = dset.index.name_to_cat['change']['id']
        for track_id, (vid_poly, score) in enumerate(scored_polys, start=1):
            for gid in gids:
                img = dset.index.imgs[gid]
                vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
                img_from_vid = vid_from_img.inv()

                # Transform the video polygon into image space
                img_poly = vid_poly.warp(img_from_vid)
                bbox = list(img_poly.bounding_box().to_coco())[0]
                # Add the polygon as an annotation on the image
                dset.add_annotation(
                    image_id=gid, category_id=change_cid,
                    bbox=bbox, segmentation=img_poly, score=score,
                    track_id=track_id)

    import ubelt as ub
    new_fpath = ub.augpath(dset.fpath, suffix='_timeagg_v1')
    dset.fpath = str(new_fpath)
    print('write dset.fpath = {!r}'.format(dset.fpath))
    dset.dump(dset.fpath, newlines=True)
    # kwplot.imshow(probs * hard_probs)


def _checkkalman():
    import simdkalman
    import einops
    import kwimage
    import numpy as np
    import kwplot
    import pandas as pd
    sns = kwplot.autosns()

    image_observations = np.stack([
        kwimage.gaussian_patch((5, 3), sigma=sigma)
        for sigma in np.random.rand(11) * 2
    ])
    observations = einops.rearrange(image_observations, 't h w -> (h w) t')
    # observations += (np.random.randn(*observations.shape) * 0.001)

    obs_df = pd.DataFrame([
        {'time': t, 'prob': v, 'track_id': track_id}
        for track_id, series in enumerate(observations)
        for t, v in enumerate(series)
    ])

    kf = simdkalman.KalmanFilter(
        state_transition=np.array([[1]]),
        process_noise=np.diag([0.1]),
        observation_model=np.array([[0.2]]),
        observation_noise=0.5)

    # smooth and explain existing data
    smoothed = kf.smooth(observations)

    smooth_df = pd.DataFrame([
        {'time': t, 'prob': v, 'track_id': track_id}
        for track_id, series in enumerate(smoothed.states.mean[:, :, 0])
        for t, v in enumerate(series)
    ])

    kwplot.figure(fnum=1, pnum=(1, 2, 1), doclf=1)
    sns.lineplot(data=obs_df, x='time', y='prob', hue='track_id')
    kwplot.figure(fnum=1, pnum=(1, 2, 2))
    sns.lineplot(data=smooth_df, x='time', y='prob', hue='track_id')

    # predict new data
    # pred = kf.predict(data, 1)
    # from filterpy import kalman

    # # Observations at time 1, 2, 3, and 4
    # Z1 = kwimage.gaussian_patch((3, 3), sigma=0.1).ravel()[None, :].T
    # Z2 = kwimage.gaussian_patch((3, 3), sigma=1.1).ravel()[None, :].T
    # Z3 = kwimage.gaussian_patch((3, 3), sigma=0.3).ravel()[None, :].T
    # Z4 = kwimage.gaussian_patch((3, 3), sigma=0.2).ravel()[None, :].T

    # P = 1 ** 2  # System variance
    # Q = 1.0     # System noise
    # P = np.full((1, 1), fill_value=1.0 ** 2)     # System variance
    # R = np.full((1, 9), fill_value=1.0 ** 2)     # Measurement variance

    # X0 = np.zeros_like(Z1)
    # X1, P1 = kalman.predict(X0, P=P, u=0, Q=Q)

    # X2, P2 = kalman.update(X1, P=P1, z=Z1, R=R)
    # pass

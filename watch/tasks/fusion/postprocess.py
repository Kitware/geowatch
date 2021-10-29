def accumulate_temporal_predictions_simple_v1(pred_fpath='/home/local/KHQ/usman.rafique/data/dvc-repos/smart_watch_dvc/training/horologic/usman.rafique/Drop1_TeamFeat_Holdout/runs/DirectCD_smt_it_joint_p8_teamfeat_v010/pred/pred.kwcoco.json'):
    """
    find $HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/ -iname "package_*.pt"

    TEST_DATASET=$HOME/remote/yardrat/smart_watch_dvc/drop1-S2-L8-aligned/vali_data.kwcoco.json

    PACKAGE_FPATH=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/checkpoints/epoch=1-step=1431.ckpt

    PACKAGE_FPATH=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/checkpoints/epoch=0-step=715-v1.ckpt



    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    TEST_DATASET=$DVC_DPATH/drop1-S2-L8-aligned/vali_data.kwcoco.json
    PACKAGE_FPATH=$DVC_DPATH/models/fusion/package_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.pt
    PRED_DATASET=./tmp_preds/tmp_pred.kwcoco.json
    python -m watch.tasks.fusion.predict \
        --test_dataset="$TEST_DATASET" \
        --package_fpath="$PACKAGE_FPATH" \
        --pred_dataset="$PRED_DATASET" \
        --write_probs=True \
        --write_preds=False --gpus=1




    DVC_DPATH=$HOME/remote/yardrat/smart_watch_dvc
    TEST_DATASET=$DVC_DPATH/drop1-S2-L8-aligned/vali_data.kwcoco.json
    PACKAGE_FPATH=$DVC_DPATH/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_0/checkpoints/epoch=2-step=2147.pt

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
    import kwcoco
    import ubelt as ub
    from watch.tasks.tracking.from_heatmap import time_aggregated_polys
    from watch.tasks.tracking.normalize import apply_tracks

    # This pred_fpath is the file is written using the model package on DVC:
    # data/dvc-repos/smart_watch_dvc/models/fusion/checkpoint_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.ckpt
    result_dataset = kwcoco.CocoDataset.coerce(ub.expandpath(pred_fpath))
    dset = result_dataset

    dset = apply_tracks(dset, time_aggregated_polys, overwrite=True)

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


if __name__ == '__main__':
    accumulate_temporal_predictions_simple_v1()

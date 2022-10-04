WORKDIR=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp
mkdir -p $WORKDIR
cd $WORKDIR
rsync -avpr horologic:/data/david.joy/Ph2Oct5/./KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco "$WORKDIR"



WORKDIR=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp
BUNDLE_DPATH=$WORKDIR/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco
TEST_DATASET=$BUNDLE_DPATH/cropped_kwcoco_for_bas.json
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
mkdir -p $BUNDLE_DPATH/testing
python -m watch.mlops.schedule_evaluation \
    --model_globstr="$MODEL_FPATH" \
    --test_dataset="$TEST_DATASET" \
    --grid_pred_pxl='
    include:
    - {
      "tta_fliprot": 0,
      "tta_time": 0,
      "chip_overlap": 0.3,
      "input_space_scale": "15GSD",
      "window_space_scale": "10GSD",
      "output_space_scale": "15GSD",
      "time_span": "auto",
      "time_sampling": "auto",
      "time_steps": "auto",
      "chip_dims": "auto",
      "set_cover_algo": "None",
      "resample_invalid_frames": 1,
      "use_cloudmask": 1
    }
    ' \
    --grid_pred_trk='
    include: 
    - {
        "thresh": 0.10,
        "morph_kernel": 3,
        "norm_ord": 1,
        "agg_fn": "probs",
        "thresh_hysteresis": None,
        "moving_window_size": None,
        "polygon_fn": "heatmaps_to_polys",
    }
    ' \
    --expt_dvc_dpath="$BUNDLE_DPATH/testing"  \
    --enable_pred_trk=1 \
    --enable_pred_pxl=1 \
    --enable_eval_pxl=1 \
    --enable_eval_trk=1 \
    --enable_pred_trk_viz=1 \
    --backend=serial --run=1  --skip_existing=1 --check_other_sessions=0




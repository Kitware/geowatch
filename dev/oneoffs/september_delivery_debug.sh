WORKDIR=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp
mkdir -p $WORKDIR
cd $WORKDIR
rsync -avpr horologic:/data/david.joy/Ph2Oct5/./KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco "$WORKDIR"



WORKDIR=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp
BUNDLE_DPATH=$WORKDIR/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco
BAS_TEST_DATASET=$BUNDLE_DPATH/cropped_kwcoco_for_bas.json
SC_TEST_DATASET=$BUNDLE_DPATH/cropped_kwcoco_for_sc.json
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
SC_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
echo "

DVC_EXPT_DPATH = $DVC_EXPT_DPATH
SC_MODEL_FPATH = $SC_MODEL_FPATH
BAS_MODEL_FPATH = $BAS_MODEL_FPATH

"
mkdir -p $BUNDLE_DPATH/testing
python -m watch.mlops.schedule_evaluation \
    --trk_model_globstr="$BAS_MODEL_FPATH" \
    --act_model_globstr="$SC_MODEL_FPATH" \
    --model_globstr="$BAS_MODEL_FPATH" \
    --trk_test_dataset="$BAS_TEST_DATASET" \
    --act_test_dataset="$SC_TEST_DATASET" \
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
    --backend=tmux --run=0  --skip_existing=1 --check_other_sessions=0




WORKDIR=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp
BUNDLE_DPATH=$WORKDIR/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco
BAS_TEST_DATASET=$BUNDLE_DPATH/cropped_kwcoco_for_bas.json
SC_TEST_DATASET=$BUNDLE_DPATH/cropped_kwcoco_for_sc.json
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
SC_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
python -m watch.mlops.schedule_evaluation \
    --params="
    - matrix:
        ###
        ### BAS Pixel Prediction
        ###
        trk.pxl.model: $BAS_MODEL_FPATH
        trk.pxl.data.test_dataset: $BAS_TEST_DATASET
        trk.pxl.data.tta_time: 0
        trk.pxl.data.chip_overlap: 0.3
        trk.pxl.data.window_scale_space: 10GSD
        trk.pxl.data.input_scale_space: 15GSD
        trk.pxl.data.output_scale_space: 15GSD
        trk.pxl.data.time_span: auto
        trk.pxl.data.time_sampling: auto
        trk.pxl.data.time_steps: auto
        trk.pxl.data.chip_dims: auto
        trk.pxl.data.set_cover_algo: None
        trk.pxl.data.resample_invalid_frames: 1
        trk.pxl.data.use_cloudmask: 1
        ###
        ### BAS Polygon Prediction
        ###
        trk.poly.thresh: 0.10
        trk.poly.morph_kernel: 3
        trk.poly.norm_ord: 1
        trk.poly.agg_fn: probs
        trk.poly.thresh_hysteresis: None
        trk.poly.moving_window_size: None
        trk.poly.polygon_fn: heatmaps_to_polys
        ###
        ### SC Pixel Prediction
        ###
        act.pxl.model: $SC_MODEL_FPATH
        act.pxl.data.test_dataset: $SC_TEST_DATASET
    " \
    --expt_dvc_dpath="$BUNDLE_DPATH/testing"  \
    --enable_pred_trk=1 \
    --enable_pred_pxl=1 \
    --enable_eval_pxl=1 \
    --enable_eval_trk=1 \
    --enable_pred_trk_viz=1 \
    --backend=tmux --run=0  --skip_existing=1 --check_other_sessions=0



python -m watch.tasks.fusion.predict \
    --write_probs=True \
    --write_preds=False \
    --with_class=auto \
    --with_saliency=auto \
    --with_change=False \
    --package_fpath=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt \
    --pred_dataset=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/testing/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco_cropped_kwcoco_for_bas/predcfg_41fd3894/pred.kwcoco.json \
    --test_dataset=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_bas.json \
    --num_workers=4 \
    --tta_fliprot=0 \
    --tta_time=0 \
    --chip_overlap=0.3 \
    --input_space_scale=15GSD \
    --window_space_scale=10GSD \
    --output_space_scale=15GSD \
    --time_span=auto \
    --time_sampling=auto \
    --time_steps=auto \
    --chip_dims=auto \
    --set_cover_algo=None \
    --resample_invalid_frames=1 \
    --use_cloudmask=1 \
    --devices=0, \
    --accelerator=gpu \
    --batch_size=1
    


python -m watch.cli.kwcoco_to_geojson \
    /home/joncrall/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/testing/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco_cropped_kwcoco_for_bas/predcfg_41fd3894/pred.kwcoco.json \
    --default_track_fn saliency_heatmaps \
    --track_kwargs '{"thresh": 0.1, "morph_kernel": 3, "norm_ord": 1, "agg_fn": "probs", "thresh_hysteresis": null, "moving_window_size": null, "polygon_fn": "heatmaps_to_polys"}' \
    --clear_annots \
    --bas_mode \
    --out_dir /home/joncrall/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/testing/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco_cropped_kwcoco_for_bas/predcfg_41fd3894/tracking/trackcfg_6db2a013/tracked_sites_bas \
    --out_fpath /home/joncrall/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/testing/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco_cropped_kwcoco_for_bas/predcfg_41fd3894/tracking/trackcfg_6db2a013/tracks_bas.json \
    --out_kwcoco /home/joncrall/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/testing/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco_cropped_kwcoco_for_bas/predcfg_41fd3894/tracking/trackcfg_6db2a013/tracks_bas.kwcoco.json


source "$HOME/code/watch/secrets/secrets"
export GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES"
export GDAL_HTTP_MULTIPLEX="YES"
export GDAL_HTTP_VERSION="2"
export VSI_CACHE="TRUE"
export VSI_CACHE_SIZE=25000000
export AWS_PROFILE=iarpa
WORKDIR=/home/joncrall/data/dvc-repos/smart_data_dvc/tmp
BUNDLE_DPATH=$WORKDIR/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco
TEST_DATASET=$BUNDLE_DPATH/cropped_kwcoco_for_bas.json
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
SC_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
mkdir -p $BUNDLE_DPATH/testing
XDEV_PROFILE=1 python -m watch.cli.coco_align_geotiffs \
    --visualize False \
    --src $BUNDLE_DPATH/kwcoco_for_sc.json \
    --dst $BUNDLE_DPATH/cropped_kwcoco_for_sc.json \
    --regions /home/joncrall/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/testing/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco_cropped_kwcoco_for_bas/predcfg_41fd3894/tracking/trackcfg_6db2a013/tracked_sites_bas/KR_R001.geojson \
    --force_nodata -9999 \
    --include_channels 'red|green|blue|cloudmask' \
    --site_summary True \
    --geo_preprop auto \
    --keep none \
    --target_gsd 4  \
    --context_factor 1.5   \
    --workers 24  \
    --aux_workers 4 \
    --rpc_align_method affine_warp  

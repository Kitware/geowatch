# Determine the paths to your SMART data and experiment repositories.
DATA_DVC_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
EXPT_DVC_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)


echo "
EXPT_DVC_DPATH=$EXPT_DVC_DPATH
DATA_DVC_DPATH=$DATA_DVC_DPATH
"


# The baseline model is checked into the experiment DVC repo.  This is the
# model we used in the November delievery. You may need to pull it from DVC if
# you haven't already.
# BASELINE_PACKAGE_FPATH="$EXPT_DVC_DPATH"/training/horologic/connor.greenwell/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_UNet/lightning_logs/version_5/checkpoints/epoch\=1092-step\=14209.pt
# BASELINE_PACKAGE_FPATH="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
# BASELINE_PACKAGE_FPATH="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
BASELINE_PACKAGE_FPATH="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305.pt


# NOTE:
# the feature_fusion_tutorial curently just runs the baseline on
# ``data_vali.kwcoco.json`` but here we run that file through ``split_videos``
# first which breaks it up into a kwcoco file per region. We then run the
# evaluation on each region separately. We will likely want to adopt this
# strategy for running evaluations so we can compare results at a more 
# granular level.
# python -m watch.cli.split_videos "$DATA_DVC_DPATH"/Drop4-BAS/data_vali.kwcoco.json

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $BASELINE_PACKAGE_FPATH
            bas_pxl.test_dataset:
                #- $DATA_DVC_DPATH/Drop4-BAS/data_vali.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R001.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R002.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/data_vali_US_R007.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/BR_R001.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/BR_R002.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/AE_R001.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_vali_US_R007_uky_invariants.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_train_BR_R001_uky_invariants.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_train_BR_R002_uky_invariants.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_train_AE_R001_uky_invariants.kwcoco.json

            bas_pxl.channels: auto
            bas_pxl.chip_dims: 128, 128
            bas_pxl.chip_overlap: 0.3
            bas_pxl.window_space_scale: auto
            bas_pxl.output_space_scale: auto
            bas_pxl.input_space_scale: auto
            bas_pxl.time_span: auto
            bas_pxl.time_sampling: auto

            bas_poly.moving_window_size: null
            bas_poly.min_area_sqkm: null
            bas_poly.max_area_sqkm: null
            bas_poly.max_area_behavior: 'ignore'
            bas_poly.response_thresh: null
            bas_poly.time_thresh: null
            bas_poly.morph_kernel: 0
            bas_poly.thresh:
                #- 0.025
                - 0.05
                #- 0.075
                - 0.1
                #- 0.125
                - 0.15
                #- 0.175
                - 0.2
                #- 0.225
                - 0.25
                #- 0.275
                - 0.3
                #- 0.325
                - 0.35
                #- 0.375
                - 0.4
                #- 0.425
                - 0.45
                #- 0.475
                - 0.5
                - 0.55
                - 0.6
                - 0.65
                - 0.7
                - 0.75
                - 0.8
                - 0.85
                - 0.9
            bas_poly.agg_fn:
                - probs
                - rescaled_probs
                #- binary
                #- rescaled_binary

            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 0
    " \
    --root_dpath="$EXPT_DVC_DPATH/_evaluations/normalized_expt2" \
    --devices="1," --queue_size=6 \
    --backend=tmux --queue_name "baseline-queue" \
    --virtualenv_cmd="conda activate watch_py3.9" \
    --pipeline=bas \
    --run=1
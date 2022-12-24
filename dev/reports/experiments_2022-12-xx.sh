# Note: Optimize Macro BAS F1 average across regions

# Real inputs, this actually will run something given the DVC repos
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

SC_MODEL=$DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
BAS_MODEL=$DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt

python -m watch.mlops.schedule_evaluation \
    --pipeline=joint_bas_sc_nocrop \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $BAS_MODEL
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
            bas_pxl.time_sampling: auto
            bas_pxl.window_space_scale: 15GSD
            bas_pxl.input_space_scale: window
            bas_poly.moving_window_size: null
            bas_poly.thresh: 0.1
            sc_pxl.window_space_scale: 8GSD
            sc_pxl.input_space_scale: window
            sc_poly.thresh: 0.1
            sc_poly.use_viterbi: 0
            sc_pxl.package_fpath: $SC_MODEL
            sc_poly_viz.enabled: 0
            sc_pxl_eval.enabled: 0
    " \
    --root_dpath=./my_dag_runs \
    --devices="0,1" --queue_size=2 --backend=serial \
    --cache=1 --skip_existing=0 --run=1


# Real data
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/bas_native_epoch44.pt
            bas_pxl.channels:
                - 'red|green|blue'
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                # - $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
            bas_pxl.chip_dims: 196,196
            bas_pxl.chip_overlap: 0.3
            bas_pxl.window_space_scale: 10GSD
            bas_pxl.output_space_scale: 10GSD
            bas_pxl.input_space_scale: native
            bas_pxl.time_span: 6m
            bas_pxl.time_sampling: soft2+distribute
            bas_poly.moving_window_size: null
            bas_poly.thresh:
                - 0.1
                - 0.2
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 1
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testpipe" \
    --devices="0,1" --queue_size=2 \
    --backend=tmux --queue_name "demo-queue" \
    --pipeline=bas \
    --run=1



# Real data
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_native_epoch44.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/BR_R002.kwcoco.json

            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt

            bas_pxl.window_space_scale:
                #- auto
                - 15GSD
                - 30GSD
            bas_pxl.input_space_scale: window
            bas_pxl.output_space_scale: window
            bas_pxl.chip_dims:
                - auto
                - 256,256
            bas_pxl.time_sampling:
                - auto
            bas_poly.moving_window_size:
                - null
                - 100
                - 200
                - 300
            bas_poly.thresh:
                - 0.1
                #- 0.13
                #- 0.2
            sc_pxl.chip_dims:
                - auto
                - 256,256
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: window
            sc_pxl.output_space_scale: window
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0
            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 1
            sc_pxl.enabled: 0
            sc_poly.enabled: 0
            sc_poly_eval.enabled: 0
            sc_pxl_eval.enabled: 0
            sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testpipe2" \
    --devices="0,1" --queue_size=2 \
    --backend=tmux \
    --pipeline=joint_bas_sc_nocrop \
    --cache=1 --skip_existing=0 \
    --rprint=1 --run=1

#--max_configs=1 \

# Real data
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_native_epoch44.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-BAS/BR_R002.kwcoco.json

            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt

            bas_pxl.window_space_scale:
                - auto
                - 15GSD
                - 30GSD
            bas_pxl.chip_dims:
                - auto
                - 256,256
            bas_pxl.time_sampling:
                - auto
            bas_pxl.input_space_scale:
                - window
            bas_poly.moving_window_size:
                - null
                - 100
                - 200
                - 300
            bas_poly.thresh:
                - 0.1
                - 0.13
                - 0.2
            sc_pxl.chip_dims:
                - auto
                - 256,256
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: window
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0

            bas_pxl.enabled: 0
            bas_poly.enabled: 1
            sc_pxl.enabled: 0
            sc_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 0
            sc_poly_eval.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testpipe" \
    --devices="0,1" --queue_size=3 \
    --backend=tmux \
    --pipeline=joint_bas_sc_nocrop \
    --cache=1 \
    --queue_name="eval-existing" \
    --run=1 --rprint=1


# Real data
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                - $DVC_EXPT_DPATH/connor-models/2022-12-19/bas_native_rgb_2022-12-19.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_native_epoch44.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/BR_R002.kwcoco.json
            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            bas_pxl.window_space_scale:
                - auto
                #- 15GSD
                #- 30GSD
            bas_pxl.chip_dims:
                - auto
                #- 256,256
            bas_pxl.time_sampling:
                - auto
            bas_pxl.input_space_scale:
                - auto
            bas_poly.moving_window_size:
                - null
                #- 100
                #- 200
                #- 300
            bas_poly.thresh:
                - 0.1
                #- 0.13
                #- 0.2
            sc_pxl.chip_dims:
                - auto
                #- 256,256
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: window
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0

            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            sc_pxl.enabled: 0
            sc_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 1
            sc_poly_eval.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_evaluations" \
    --devices="0," --queue_size=1 \
    --backend=tmux \
    --pipeline=bas --rprint=1 \
    --cache=1 \
    --queue_name="eval-models" \
    --run=0


#####################
## SC-Only Evaluation 
#####################
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

# Dataset munging
# ---------------
if [ ! -f "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R001_sites.kwcoco.json ]; then
    # Split into a kwcoco file per site
    python -m watch.cli.split_videos "$DVC_DATA_DPATH"/Drop4-SC/data_vali.kwcoco.json
    # Combine sites per region
    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R001_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R001_sites.kwcoco.json
    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R002_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R002_sites.kwcoco.json
    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_vali_US_R007_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_vali_US_R007_sites.kwcoco.json
    # Remove the temporary split sites
    rm "$DVC_DATA_DPATH"/Drop4-SC/data_vali_*_box.kwcoco.json
fi

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
                - $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=35-step=486072.pt
            sc_pxl.test_dataset:
                #- $DVC_DATA_DPATH/Drop4-SC/data_vali.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-SC/data_vali_KR_R001_sites.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-SC/data_vali_KR_R002_sites.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-SC/data_vali_US_R007_sites.kwcoco.json
            sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/region_models
            sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/site_models
            sc_pxl.chip_dims:
                - auto
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: auto
            sc_pxl.output_space_scale: auto
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0
            sc_pxl.enabled: 1
            sc_poly.enabled: 1
            sc_poly_eval.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly_viz.enabled: 1
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testsc" \
    --devices="0," --queue_size=1 \
    --backend=tmux \
    --pipeline=sc \
    --cache=1 --skip_existing=0 \
    --rprint=1 --run=0

#!/bin/bash

export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase3_data")
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase3_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "status" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "list checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "repackage checkpoints" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "gather packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop8-ARA-Median10GSD-V1"


# Point based mlops evaluation
export DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
export DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop8-v1
MLOPS_NAME=_preeval23_point_bas_grid
MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_NAME

#TODO: perform query showing only v4 models AND last model in list (aka baseline)
#      Avoid jank way using agregate hundreds of times
MODEL_SHORTLIST="
- /home/local/KHQ/vincenzo.dimatteo/Desktop/dvc_repos/smart_phase3_expt/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4_epoch128_step10836.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4_epoch117_step9912.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4_epoch129_step10920.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4_epoch165_step13944.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4_epoch131_step11088.pt
- /home/local/KHQ/vincenzo.dimatteo/Desktop/dvc_repos/smart_phase3_expt//models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch56_step4788.pt
"

mkdir -p "$MLOPS_DPATH"
echo "$MODEL_SHORTLIST" > "$MLOPS_DPATH/shortlist.yaml"

cat "$MLOPS_DPATH/shortlist.yaml"

# FIXME: make sdvc request works with YAML lists of models
# sdvc request --verbose=3 "$MLOPS_DPATH/shortlist.yaml"

geowatch schedule --params="
    pipeline: bas

    matrix:
        bas_pxl.package_fpath: $MLOPS_DPATH/shortlist.yaml

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/CN_C000/imganns-CN_C000-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/KW_C001/imganns-KW_C001-rawbands.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/CO_C001/imganns-CO_C001-rawbands.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.36
            #- 0.37
            - 0.375
            #- 0.38
            - 0.39
            - 0.40
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.85
            #- 0.8
            - 0.75
        bas_poly.time_pad_after:
            #- 0 months
            - 3 months
            #- 12 months
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $TRUTH_DPATH/region_models
        bas_poly_eval.true_site_dpath: $TRUTH_DPATH/site_models
        bas_poly_eval.true_region_dpath: $TRUTH_DPATH/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly_viz.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
    include:
        - bas_poly.time_pad_after: 0 months
          bas_poly.time_pad_before: 0 months
        - bas_poly.time_pad_after: 3 months
          bas_poly.time_pad_before: 3 months
        - bas_poly.time_pad_after: 12 months
          bas_poly.time_pad_before: 12 months
    " \
    --root_dpath="$MLOPS_DPATH" \
    --devices="1,2,3" --tmux_workers=8 \
    --backend=tmux --queue_name "$MLOPS_NAME" \
    --skip_existing=1 \
    --run=1

# Point based mlops evaluation
export DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
export DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop8-v1
MLOPS_NAME=_preeval23_point_bas_grid
MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_NAME
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m geowatch.mlops.aggregate \
    --embed \
    --pipeline=bas \
    --target "
        - $MLOPS_DPATH
        - $DVC_EXPT_DPATH/_preeval22_point_bas_grid
    " \
    --output_dpath="$MLOPS_DPATH/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.bas_poly.thresh
            - resolved_params.bas_pxl.package_fpath
            - resolved_params.bas_pxl.channels
    " \
    --stdout_report="
        top_k: 100000000000
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    "

export DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
export DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop8-v1
MLOPS_NAME=_preeval23_point_bas_grid
MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_NAME
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m geowatch.mlops.aggregate \
    --embed \
    --pipeline=bas \
    --target "
        - $MLOPS_DPATH
        - $DVC_EXPT_DPATH/_preeval22_point_bas_grid
    " \
    --output_dpath="$MLOPS_DPATH/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.bas_poly.thresh
            - resolved_params.bas_pxl.package_fpath
            - resolved_params.bas_pxl.channels
    " \
    --stdout_report="
        top_k: 100000000000
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --rois="KR_R002,CN_C000,KW_C001"\
    --query="df.table['params.bas_pxl.package_fpath'].str.contains('$DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4/Drop8_ARA_Median10GSD_allsensors_pointannsv1_fixeddataloader_v4_epoch131_step11088.pt') | df.table['params.bas_pxl.package_fpath'].str.contains('/home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Median10GSD-V1/packages/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1/Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1_epoch56_step4788.pt')"

    #--rois="KR_R002,CN_C000,KW_C001"
    #--rois="KR_R002"
    #--rois="KR_R002,CN_C000,KW_C001,CO_C001"
    #" --rois="KR_R002,CN_C000"
    #--rois="CN_C000"

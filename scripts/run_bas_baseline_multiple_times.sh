#!/bin/bash
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)

geowatch schedule --params="
    pipeline: bas

    matrix:

        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/KR_R002/imgonly-KR_R002-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/KW_C001/imgonly-KW_C001-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CO_C001/imgonly-CO_C001-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CN_C000/imgonly-CN_C000-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/NZ_R001/imgonly-NZ_R001-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CH_R001/imgonly-CH_R001-rawbands.kwcoco.zip

        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.37
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_bas_only_baseline_${HOSTNAME}_v5" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_bas_only_baseline_${HOSTNAME}_v5" \
    --skip_existing=1 \
    --run=1


#python -m watch.mlops.aggregate \
#    /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/__bas_only_baseline_multi/aggregate/bas_poly_eval_toothbrush_00018_2023-10-10T133732-5.csv.zip \
#    /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/__bas_only_baseline_multi/aggregate/bas_poly_eval_namek_00012_2023-10-10T133729-5.csv.zip


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas \
    --export_tables=True \
    --target "
        - $DVC_EXPT_DPATH/_bas_only_baseline_${HOSTNAME}_v1
        - $DVC_EXPT_DPATH/_bas_only_baseline_${HOSTNAME}_v2
        - $DVC_EXPT_DPATH/_bas_only_baseline_${HOSTNAME}_v3
        - $DVC_EXPT_DPATH/_bas_only_baseline_${HOSTNAME}_v4
        - $DVC_EXPT_DPATH/_bas_only_baseline_${HOSTNAME}_v5
    " \
    --output_dpath="$DVC_EXPT_DPATH/__bas_only_baseline_multi/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 11
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 0
        show_csv: 0
    "

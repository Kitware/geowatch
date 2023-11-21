
# Eval11 Baseline on Drop7-MedianNoWinter10GSD
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - [196,196]
        bas_pxl.time_span:
            - auto
        bas_pxl.input_space_scale:
            - 10GSD
        bas_pxl.time_sampling:
            - soft4
        bas_poly.thresh:
            - 0.4
            - 0.425
            - 0.45
        bas_poly.time_thresh:
            - 0.8
        bas_poly.inner_window_size:
            - 1y
        bas_poly.inner_agg_fn:
            - max
        bas_poly.norm_ord:
            - inf
        bas_poly.moving_window_size:
            - null
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0

    submatrices:
        - bas_pxl.input_space_scale: 10GSD
          bas_pxl.window_space_scale: 10GSD
          bas_pxl.output_space_scale: 10GSD
          bas_poly.resolution: 10GSD
    " \
    --root_dpath="$DVC_EXPT_DPATH/_drop7_nowinter_baseline" \
    --devices="0,1" --tmux_workers=6 \
    --backend=tmux --queue_name "_drop7_nowinter_baseline" \
    --pipeline=bas --skip_existing=1 \
    --run=1


# Pull out baseline tables
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_drop7_nowinter_baseline
    " \
    --output_dpath="$DVC_EXPT_DPATH/_drop7_nowinter_baseline/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
    " \
    --plot_params="
        enabled: 0
    " \
    --stdout_report="
        top_k: 5
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: False
        reference_region: final
    " \
    --rois="auto"

    --query='
        (df["params.bas_poly.thresh"] == 0.425) &
        (df["params.bas_poly.time_thresh"] == 0.8) &
        (df["params.bas_pxl.chip_dims"].apply(str).str.contains("196")) &
        (df["params.bas_pxl.package_fpath"].str.contains("Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026"))
    '


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD


sdvc request "
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch=189-step=12160-val_loss=2.881.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch196_step6304.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_split6_V87/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_split6_V87_epoch43_step1408.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch10_step176.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch5_step96.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch521_step22446.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch0_step86.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch334_step10720.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch333_step10688.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch337_step10816.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch329_step10560.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch273_step8768.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch343_step11008.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch112_step904.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch106_step856.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch12_step104.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch26_step864.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch163_step5248.pt.dvc
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=0-step=5778-v1.pt.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=1-step=11556.pt.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334-v1.pt.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch1_step11556.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch2_step17334.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch3_step22551.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v1_epoch0_step5778.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/package_epoch3_step22551.pt.pt
" --verbose

python -m watch.mlops.manager "list packages" --dataset_codes Drop7-MedianNoWinter10GSD-NoMask --yes
python -m watch.mlops.manager "list packages" --dataset_codes Drop7-Cropped2GSD


python -m watch.mlops.schedule_evaluation --params="
    pipeline: sc

    matrix:
        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.test_dataset:
            - $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands-small.kwcoco.zip
            - $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip
            - $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands-small.kwcoco.zip
            - $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands-small.kwcoco.zip

        sc_pxl.package_fpath:
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch=189-step=12160-val_loss=2.881.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch196_step6304.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_split6_V87/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_split6_V87_epoch43_step1408.pt

            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch521_step22446.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch0_step86.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch334_step10720.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch333_step10688.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch337_step10816.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch329_step10560.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_4GSD_split6_V88_epoch273_step8768.pt

            #- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch10_step176.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch5_step96.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch343_step11008.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch112_step904.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch26_step864.pt

            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch106_step856.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch12_step104.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch163_step5248.pt

            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=0-step=5778-v1.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=1-step=11556.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334-v1.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch1_step11556.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch2_step17334.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch3_step22551.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v1_epoch0_step5778.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/package_epoch3_step22551.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch26_step864.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch109_step3520.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch63_step2048.pt


        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3

        sc_pxl.fixed_resolution:
            - 8GSD
            #- 4GSD
            #- 2GSD

        sc_pxl.set_cover_algo: null
        sc_pxl.resample_invalid_frames: 3
        sc_pxl.observable_threshold: 0.0
        sc_pxl.mask_low_quality: true
        sc_pxl.drop_unused_frames: true
        sc_pxl.num_workers: 12
        sc_pxl.batch_size: 1
        sc_pxl.write_workers: 0

        ########################
        ## AC/SC POLY PARAMS  ##
        ########################

        sc_poly.thresh:
         - 0.07
         - 0.10
         - 0.20
         - 0.30
        sc_poly.boundaries_as: polys
        sc_poly.min_area_square_meters: 7200
        sc_poly.resolution: 8GSD

        sc_poly.site_score_thresh:
            - null
            - 0.0
            - 0.10
            - 0.30
            - 0.40
            #- 0.45
            #- 0.5

        sc_poly.smoothing:
            - null
            - 0.66

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $BUNDLE_DPATH/bas_small_truth/site_models
        sc_poly_eval.true_region_dpath: $BUNDLE_DPATH/bas_small_truth/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 0
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0

    submatrices:

        # Point each region to the polygons that AC/SC will score

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/KR_R002.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/KW_C501.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/CO_C501.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/CN_C500.geojson

    " \
    --root_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_namek_v1" \
    --queue_name "_ac_static_small_baseline_namek_v1" \
    --devices="0,1" \
    --backend=tmux --tmux_workers=8 \
    --cache=1 --skip_existing=1 --run=1


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m watch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_namek_v1
    " \
    --output_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_namek_v1/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sc_poly.thresh
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002,KW_C501,CO_C501,CN_C500"

    #--query="df['params.sc_pxl.package_fpath'].str.contains('Drop4_tune_V30_8GSD_V3_epoch=2-step=17334') & (df['params.sc_pxl.fixed_resolution'] == '8GSD')"
    #--rois="KR_R002,KW_C001,CO_C001,CN_C000" \
    #--query="df['param_hashid'] == 'lefmkkfomcev'" \
    #--rois="KR_R002"

#--query="df['param_hashid'] == 'lefmkkfomcev'"




# @4GSD
#'nkoggbaiufwf': {
#    'params.sc_poly.thresh': 0.2,
#    'params.sc_poly.site_score_thresh': 0.45,
#    'params.sc_pxl.package_fpath': 'Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch521_step22446',
#},
#'uahkqdbwwqqj': {
#    'params.sc_poly.thresh': 0.1,
#    'params.sc_poly.site_score_thresh': float('nan'),
#    'params.sc_pxl.package_fpath': 'Drop4_tune_V30_8GSD_V3_epoch=2-step=17334',
#},
#'cgqlkwevlhri': {
#    'params.sc_poly.thresh': 0.3,
#    'params.sc_poly.site_score_thresh': float('nan'),
#    'params.sc_pxl.package_fpath': 'Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch10_step176',
#},
#'ozrionpgkmty': {
#    'params.sc_poly.thresh': 0.3,
#    'params.sc_poly.site_score_thresh': float('nan'),
#    'params.sc_pxl.package_fpath': 'Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch5_step96',
#},
#'jyhiamebdzjl': {
#    'params.sc_poly.thresh': 0.3,
#    'params.sc_poly.site_score_thresh': float('nan'),
#    'params.sc_pxl.package_fpath': 'Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907',
#},
#models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
#
#
##
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch0_step16.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch1_step32.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch2_step48.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch326_step10464.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch332_step10656.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch333_step10688.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch338_step10848.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90/Drop7-Cropped2GSD_SC_bgrn_gnt_4GSD_split6_V90_epoch343_step11008.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/.gitignore
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch0_step8.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch108_step872.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch112_step904.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch12_step104.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch18_step152.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch1_step16.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch3_step32.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch4_step40.pt.dvc
# create mode 100644 models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89/Drop7-Cropped2GSD_SC_bgrn_sgd_gnt_8GSD_split6_V89_epoch79_step640.pt.dvc
##
#


#models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch26_step864.pt.dvc
#
#models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch109_step3520.pt.dvc
#models/fusion/Drop7-Cropped2GSD-V2/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91/Drop7-Cropped2GSD_SC_bgrn_gnt_2GSD_split6_V91_epoch63_step2048.pt.dvc

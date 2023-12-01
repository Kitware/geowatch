#!/bin/bash
#
# Build a static version that contains all of the predicted regions from BAS
# This will be used for AC/SC model selection

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)

sdvc request "
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
" --verbose

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
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/NZ_R001/imgonly-NZ_R001-rawbands.kwcoco.zip

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
    --root_dpath="$DVC_EXPT_DPATH/_toothbrush_for_scac_vali_dataset" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_toothbrush_for_scac_vali_dataset" \
    --skip_existing=1 \
    --run=1


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_toothbrush_for_scac_vali_dataset
    " \
    --output_dpath="$DVC_EXPT_DPATH/_toothbrush_for_scac_vali_dataset/aggregate" \
    --resource_report=0 \
    --rois=KR_R002,KW_C001,CO_C001,CN_C000 \
    --stdout_report="
        top_k: 3
        analyze: 0
        concise: 1
    "



# Gather the BAS predicted site summaries and cluster them.
python -c "if 1:
    import watch
    dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt')
    dvc_data_dpath = watch.find_dvc_dpath(tags='drop7_data')
    bas_sitesum_fpaths = list(dvc_expt_dpath.glob('_toothbrush_for_scac_vali_dataset/pred/flat/bas_poly/*/site_summaries/*'))
    bas_sites_fpaths = list(dvc_expt_dpath.glob('_toothbrush_for_scac_vali_dataset/pred/flat/bas_poly/*/sites/*'))

    # ub.udict(ub.group_items(bas_sites_fpaths, key=lambda x: x.name.split('_')[0])).map_values(len)

    new_bundle_dpath = dvc_data_dpath / 'Drop7-StaticACTestSet-2GSD'
    bas_output_dpath = new_bundle_dpath / 'bas_output'
    bas_output_dpath.ensuredir()

    pred_rm_dpath = (bas_output_dpath / 'region_models').ensuredir()
    pred_sm_dpath = (bas_output_dpath / 'site_models').ensuredir()

    for p in bas_sites_fpaths:
        new_fpath = pred_sm_dpath / p.name
        if not new_fpath.exists():
            p.copy(new_fpath)

    for p in bas_sitesum_fpaths:
        new_fpath = pred_rm_dpath / p.name
        if not new_fpath.exists():
            p.copy(new_fpath)
"


SRC_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=ssd)
export SRC_DVC_DATA_DPATH
export DST_DVC_DATA_DPATH
echo "SRC_DVC_DATA_DPATH = $SRC_DVC_DATA_DPATH"
echo "DST_DVC_DATA_DPATH = $DST_DVC_DATA_DPATH"

# Cluster BAS Predictions
REGION_IDS_STR=$(python -c "if 1:
    import pathlib
    import os
    SRC_DVC_DATA_DPATH = pathlib.Path(os.environ.get('SRC_DVC_DATA_DPATH'))
    DST_DVC_DATA_DPATH = pathlib.Path(os.environ.get('DST_DVC_DATA_DPATH'))

    src_bundle = SRC_DVC_DATA_DPATH / 'Aligned-Drop7'

    region_dpath = DST_DVC_DATA_DPATH / 'Drop7-StaticACTestSet-2GSD/bas_output/region_models'

    region_fpaths = list(region_dpath.glob('*_[RC]*.geojson'))
    region_names = [p.stem for p in region_fpaths]
    final_names = []
    for region_name in region_names:
        coco_fpath = src_bundle / region_name / f'imgonly-{region_name}-rawbands.kwcoco.zip'
        if coco_fpath.exists():
            final_names.append(region_name)
    print(' '.join(sorted(final_names)))
    ")
#REGION_IDS_STR="CN_C000 KW_C001 SA_C001 CO_C001 VN_C002"

echo "REGION_IDS_STR = $REGION_IDS_STR"

# shellcheck disable=SC2206
REGION_IDS_ARR=($REGION_IDS_STR)

REGION_IDS_ARR=(NZ_R001)

for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
done


SRC_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=ssd)
TRUE_REGION_DPATH=$DST_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_output/region_models
SRC_BUNDLE_DPATH=$SRC_DVC_DATA_DPATH/Aligned-Drop7
DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

### Cluster and Crop Jobs
python -m cmd_queue new "crop_for_sc_test_queue"
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    REGION_GEOJSON_FPATH=$TRUE_REGION_DPATH/$REGION_ID.geojson
    REGION_CLUSTER_DPATH=$DST_BUNDLE_DPATH/$REGION_ID/clusters
    SRC_KWCOCO_FPATH=$SRC_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
    DST_KWCOCO_FPATH=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
    if ! test -f "$DST_KWCOCO_FPATH"; then
        cmd_queue submit --jobname="cluster-$REGION_ID" --depends="None" -- crop_for_sc_test_queue \
            python -m geowatch.cli.cluster_sites \
                --src "$REGION_GEOJSON_FPATH" \
                --minimum_size "256x256@2GSD" \
                --dst_dpath "$REGION_CLUSTER_DPATH" \
                --draw_clusters True

        python -m cmd_queue submit --jobname="crop-$REGION_ID" --depends="cluster-$REGION_ID" -- crop_for_sc_test_queue \
            python -m geowatch.cli.coco_align \
                --src "$SRC_KWCOCO_FPATH" \
                --dst "$DST_KWCOCO_FPATH" \
                --regions "$REGION_CLUSTER_DPATH/*.geojson" \
                --rpc_align_method orthorectify \
                --workers=10 \
                --aux_workers=2 \
                --force_nodata=-9999 \
                --context_factor=1.0 \
                --minimum_size="256x256@2GSD" \
                --force_min_gsd=2.0 \
                --convexify_regions=True \
                --target_gsd=2.0 \
                --geo_preprop=False \
                --exclude_sensors=L8 \
                --sensor_to_time_window "
                    S2: 1month
                " \
                --keep img
    fi
done
python -m cmd_queue show "crop_for_sc_test_queue"
python -m cmd_queue run --workers=8 "crop_for_sc_test_queue"


# Filter out bad images
DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=ssd)
DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD
python -m cmd_queue new "filter_bad_queue"
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    KWCOCO_FPATH1=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
    KWCOCO_FPATH2=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands-filtered.kwcoco.zip
    cmd_queue submit --jobname="filter-imgs-$REGION_ID" --depends="None" -- filter_bad_queue \
        # FIXME
        #python -m geowatch.cli.cluster_sites \
        #    --src "$KWCOCO_FPATH1" \
        #    --dst "$KWCOCO_FPATH2" \
        #    --delete_assets=False \
        #    --overview=0 \
        #    --interactive=False

done
python -m cmd_queue show "filter_bad_queue"
python -m cmd_queue run --workers=8 "filter_bad_queue"


# Overwrite unfiltered inplace
DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=ssd)
DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD
python -m cmd_queue new "overwrite_with_filter_queue"
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    KWCOCO_FPATH1=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
    KWCOCO_FPATH2=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands-filtered.kwcoco.zip
    cmd_queue submit --jobname="filter-imgs-$REGION_ID" --depends="None" -- overwrite_with_filter_queue \
        mv "$KWCOCO_FPATH2" "$KWCOCO_FPATH1"

done

cd "$DST_BUNDLE_DPATH"
dvc unprotect -- */*.kwcoco.zip
ls -- */*.kwcoco.zip
python -m cmd_queue show "overwrite_with_filter_queue"
python -m cmd_queue run --backend=serial --workers=8 "overwrite_with_filter_queue"




######
# Filter to spatial subset
# See
python ./filter_sc_baseline_to_crops.py



DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=ssd)
DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD
cd "$DST_BUNDLE_DPATH"

dvc add -vvv -- \
    bas_output \
    */imgonly-*-rawbands.kwcoco.zip \
    */*/S2 \
    */*/WV \
    */*/WV1 && \
git commit -m "Update Drop7-StaticACTestSet-2GSD" && \
git push && \
dvc push -r aws -R . -vvv


dvc add -vvv -- \
    bas_small_truth \
    */imgonly-*-rawbands.kwcoco.zip \
    */imgonly-*-rawbands-small.kwcoco.zip && \
git commit -m "Update Drop7-StaticACTestSet-2GSD" && \
git push && \
dvc push -r aws -R . -vvv

#### Small Baseline Evaluation


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m geowatch.mlops.schedule_evaluation --params="
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
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt

        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3

        sc_pxl.fixed_resolution:
            - 8GSD

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
         - 0.10
        sc_poly.boundaries_as: polys
        sc_poly.min_area_square_meters: 7200
        sc_poly.resolution: 8GSD

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
    --root_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_v1" \
    --queue_name "_ac_static_small_baseline_v1" \
    --devices="0,1" \
    --backend=tmux --tmux_workers=4 \
    --cache=1 --skip_existing=1 --run=1


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
TRUTH_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_v1
    " \
    --output_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_v1/aggregate" \
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
        top_k: 100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002,KW_C501,CO_C501,CN_C500"


#### Small Extended Baseline Evaluation


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m geowatch.mlops.schedule_evaluation --params="
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
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch=189-step=12160-val_loss=2.881.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt

        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3

        sc_pxl.fixed_resolution:
            - 8GSD
            - 4GSD
            - 2GSD

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
    --root_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_v2" \
    --queue_name "_ac_static_small_baseline_${HOSTNAME}_v2" \
    --devices="0,1" \
    --backend=tmux --tmux_workers=4 \
    --cache=1 --skip_existing=1 --run=1


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
TRUTH_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_v1
    " \
    --output_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_v1/aggregate" \
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
        top_k: 100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002,KW_C501,CO_C501,CN_C500"



#### Extended Evaluation


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
TRUTH_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m geowatch.mlops.schedule_evaluation --params="
    pipeline: sc

    matrix:
        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.test_dataset:
            - $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands.kwcoco.zip
            - $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands.kwcoco.zip
            - $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands.kwcoco.zip
            - $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands.kwcoco.zip

        sc_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch=189-step=12160-val_loss=2.881.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt

        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3

        sc_pxl.fixed_resolution:
            #- 8GSD
            #- 2GSD
            - 2GSD

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
        #sc_poly.resolution: 2GSD
        sc_poly.min_area_square_meters: 7200

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $TRUTH_DVC_DATA_DPATH/annotations/drop7-hard-v1/site_models
        sc_poly_eval.true_region_dpath: $TRUTH_DVC_DATA_DPATH/annotations/drop7-hard-v1/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 0
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0

    submatrices:

        # Point to the polygons that AC/SC will score
        # Might abstract this for convinience later.

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KR_R001/imgonly-KR_R001-rawbands.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_output/region_models/KR_R001.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_output/region_models/KR_R002.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CH_R001/imgonly-CH_R001-rawbands.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_output/region_models/CH_R001.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_output/region_models/KW_C001.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_output/region_models/CO_C001.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_output/region_models/CN_C000.geojson

    " \
    --root_dpath="$DVC_EXPT_DPATH/_ac_static_baseline" \
    --queue_name "_ac_static_baseline" \
    --devices="0,1" \
    --backend=tmux --tmux_workers=4 \
    --cache=1 --skip_existing=1 --run=1


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
TRUTH_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_baseline
    " \
    --output_dpath="$DVC_EXPT_DPATH/_ac_static_baseline/aggregate" \
    --resource_report=1 \
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
        top_k: 100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002"

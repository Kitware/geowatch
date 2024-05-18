# SeeAlso
# ~/code/geowatch/docs/source/manual/baselines/baseline-2023-06-22-sc_truth.rst
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)


MLOPS_DPATH=$DVC_EXPT_DPATH/_drop8_ara_sc_v1
MODEL_SHORTLIST="
- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
- $DVC_EXPT_DPATH/models/fusion/dzyne/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_100pctphase.pt
- $DVC_EXPT_DPATH/models/fusion/dzyne/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_85pctphase.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch0_step169.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch11_step2028.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch12_step2197.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch1_step338.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch2_step507.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch3_step676.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch43_step7436.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch5_step1014.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch10_step1859.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch18_step3211.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch24_step4225.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch25_step4394.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch26_step4563.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch27_step4732.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch28_step4901.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch32_step5577.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch33_step5746.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch34_step5915.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch35_step6084.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch36_step6253.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch37_step6422.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch38_step6591.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch3_step676.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v2_V003_epoch9_step1690.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch3_step1368.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch6_step2394.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch4_step1710.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch78_step27018.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch359_step122881.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch2_step1026.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch70_step24282.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch142_step48906.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch104_step35910.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch5_step2052.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002_epoch113_step38988.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch0_step21021.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch1_step42042.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch3_step84084.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch2_step63063.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch4_step105105.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch5_step122881.pt
"


MODEL_SHORTLIST="
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch0_step169.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch11_step2028.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch12_step2197.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch1_step338.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch2_step507.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_from_v1_V002_epoch3_step676.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/dzyne/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_100pctphase.pt
- /home/local/KHQ/jon.crall/data/dvc-repos/smart_phase3_expt/models/fusion/dzyne/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_85pctphase.pt
"


mkdir -p "$MLOPS_DPATH"
echo "$MODEL_SHORTLIST" > "$MLOPS_DPATH/ac_model_shortlist.yaml"
cat "$MLOPS_DPATH/ac_model_shortlist.yaml"

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
MLOPS_DNAME=_drop8_ara_sc_v1
MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_DNAME
python -m geowatch.mlops.schedule_evaluation --params="
    pipeline: sc
    matrix:
        #####################
        ## AC PIXEL PARAMS ##
        #####################
        sc_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop8-ARA-Cropped2GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip

        sc_pxl.package_fpath:
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
            #- $DVC_EXPT_DPATH/models/fusion/dzyne/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_100pctphase.pt
            #- $DVC_EXPT_DPATH/models/fusion/dzyne/Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_85pctphase.pt
            - $MLOPS_DPATH/ac_model_shortlist.yaml

        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap:
            - 0.3
            - 0.0
        sc_pxl.num_workers: 4
        sc_pxl.batch_size: 1
        sc_pxl.write_workers: 0
        sc_pxl.observable_threshold: 0.0
        sc_pxl.drop_unused_frames: true

        sc_pxl.fixed_resolution:
            - 8GSD
            - 6GSD
            - 4GSD
        #####################
        ## SC POLY PARAMS  ##
        #####################

        sc_poly.boundaries_as:
            - bounds
            - poly
        sc_poly.new_algo: crall
        sc_poly.polygon_simplify_tolerance: 2
        sc_poly.site_score_thresh: 0.3
        sc_poly.smoothing: 0.0
        sc_poly.thresh:
            - 0.07
            - 0.2
            - 0.25
            - 0.3
        sc_poly.resolution: 8GSD
        sc_poly.min_area_square_meters: 7200

        ##########################
        ## SC POLY EVAL PARAMS  ##
        ##########################

        sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop8-v1/site_models
        sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop8-v1/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 0
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0

    submatrices:
        - sc_pxl.input_space_scale: 8GSD
          sc_pxl.window_space_scale: 8GSD
          sc_pxl.output_space_scale: 8GSD
          sc_pxl.fixed_resolution: 8GSD
        - sc_pxl.input_space_scale: 2GSD
          sc_pxl.window_space_scale: 2GSD
          sc_pxl.output_space_scale: 2GSD
          sc_pxl.fixed_resolution: 2GSD
        - sc_pxl.input_space_scale: 4GSD
          sc_pxl.window_space_scale: 4GSD
          sc_pxl.output_space_scale: 4GSD
          sc_pxl.fixed_resolution: 4GSD
        - sc_pxl.input_space_scale: 6GSD
          sc_pxl.window_space_scale: 6GSD
          sc_pxl.output_space_scale: 6GSD
          sc_pxl.fixed_resolution: 6GSD

    submatrices1:
        # Because there is no BAS component, we need to provide site summaries.
        # We should use BAS system outputs, but if we dont have those use truth.
        - sc_pxl.test_dataset: $DVC_DATA_DPATH/Drop8-ARA-Cropped2GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip
          sc_poly.site_summary: $DVC_DATA_DPATH/annotations/drop8-v1/region_models/KR_R002.geojson

        #- sc_pxl.test_dataset: $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip
        #  sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/KW_C501.geojson
    " \
    --root_dpath="$MLOPS_DPATH" \
    --queue_name "$MLOPS_DNAME" \
    --devices="0,1,2,3" \
    --backend=tmux --tmux_workers=4 \
    --cache=1 --skip_existing=1 --run=1


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
MLOPS_DNAME=_drop8_ara_sc_v1
MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_DNAME
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m geowatch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $MLOPS_DPATH
    " \
    --output_dpath="$MLOPS_DPATH/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
    " \
    --plot_params="
        enabled: 1
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sc_poly.thresh
            - params.sc_poly.boundaries_as
            - params.sc_pxl.fixed_resolution
            - params.sc_pxl.chip_overlap
            - params.sc_pxl.package_fpath
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


#sc_pxl.tta_fliprot: 0.0
#sc_pxl.tta_time: 0.0
#sc_pxl.chip_overlap: 0.3
#sc_pxl.input_space_scale: 8GSD
#sc_pxl.window_space_scale: 8GSD
#sc_pxl.output_space_scale: 8GSD
#sc_pxl.time_span: 6m
#sc_pxl.time_sampling: auto
#sc_pxl.time_steps: 12
#sc_pxl.chip_dims: auto
#sc_pxl.set_cover_algo: null
#sc_pxl.resample_invalid_frames: 3
#sc_pxl.observable_threshold: 0.0
#sc_pxl.mask_low_quality: true
#sc_pxl.drop_unused_frames: true
#sc_poly.thresh: 0.07
#sc_poly.boundaries_as: polys
#sc_poly.resolution: 8GSD
#sc_poly.min_area_square_meters: 7200

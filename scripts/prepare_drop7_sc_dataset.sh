#!/bin/bash
# To build this dataset we will use predictions from our BAS model to enrich
# the dataset with additional negatives.


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)

# List the regions we will predict on
# ls -- "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-NoMask/*/imganns-*[0-9].kwcoco.zip
python -c "if 1:
    import ubelt as ub

    from simple_dvc import registery
    dvc_dpath = registery.find_dvc_dpath(tags='phase2_data', hardware='ssd')

    from kwutil.util_yaml import Yaml
    keyname = chr(36) + 'DVC_DATA_DPATH'
    dvc_dpath = ub.Path('$DVC_DATA_DPATH').expand()
    paths = list(dvc_dpath.glob('Drop7-MedianNoWinter10GSD-NoMask/*/imganns-*[0-9].kwcoco.zip'))
    print(Yaml.dumps([keyname + '/' + str(p.relative_to(dvc_dpath)).replace( for p in paths]))
    "


ACSC_MODELS="
    - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch186_step16082.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch57_step19836.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch41_step3612.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch173_step14964.pt
    "

_="
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch60_step5246.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch78_step6794.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch77_step6708.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch75_step6536.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch48_step4214.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch2_step258.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch74_step6450.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch1_step172.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch0_step86.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch84_step7310.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch54_step4730.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch104_step9030.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch95_step8256.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch103_step8944.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch311_step26832.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch236_step20382.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch1_step172.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch0_step86.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch230_step19866.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch240_step20726.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch250_step21586.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch374_step16125.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch521_step22446.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch129_step5590.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch272_step11739.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch217_step9374.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch518_step22317.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch165_step14276.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch138_step11954.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch148_step12814.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch185_step15996.pt
"

BAS_MODELS="
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
"


#TEST_DATASETS="
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R002/imganns-KR_R002.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CN_C000/imganns-CN_C000.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R002/imganns-BR_R002.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CH_R001/imganns-CH_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/NZ_R001/imganns-NZ_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/AE_R001/imganns-AE_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BH_R001/imganns-BH_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R001/imganns-BR_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R004/imganns-BR_R004.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R005/imganns-BR_R005.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R001/imganns-KR_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/LT_R001/imganns-LT_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/PE_R001/imganns-PE_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/US_R001/imganns-US_R001.kwcoco.zip
#    - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/US_R004/imganns-US_R004.kwcoco.zip
#"


OTHER_MODELS="
    - $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
    - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
"

#sdvc request ""
#python -c "import sys; [print(v) for v in sys.argv]" "
sdvc request "
$ACSC_MODELS
$BAS_MODELS
$OTHER_MODELS
" --verbose

geowatch schedule --params="
    pipeline: full

    # Convinience argument
    smart_highres_bundle: $DVC_HDD_DATA_DPATH/Aligned-Drop7

    matrix:

        sc_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch186_step16082.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch57_step19836.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch173_step14964.pt

        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R002/imganns-KR_R002.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CN_C000/imganns-CN_C000.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R002/imganns-BR_R002.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CH_R001/imganns-CH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/NZ_R001/imganns-NZ_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/AE_R001/imganns-AE_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BH_R001/imganns-BH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R001/imganns-BR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R004/imganns-BR_R004.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R005/imganns-BR_R005.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R001/imganns-KR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/LT_R001/imganns-LT_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/PE_R001/imganns-PE_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/US_R001/imganns-US_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/US_R004/imganns-US_R004.kwcoco.zip


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
            #- 0.6
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

        ######################
        ## SV Params Params ##
        ######################
        sv_crop.enabled: 0
        sv_crop.minimum_size: '256x256@2GSD'
        sv_crop.num_start_frames: 3
        sv_crop.num_end_frames: 3
        sv_crop.context_factor: 1.6

        sv_dino_boxes.enabled: 0
        sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
        sv_dino_boxes.window_dims: 256
        sv_dino_boxes.window_overlap: 0.5
        sv_dino_boxes.fixed_resolution: 3GSD

        sv_dino_filter.enabled: 0
        sv_dino_filter.end_min_score:
            - 0.15
        sv_dino_filter.start_max_score: 1.0
        sv_dino_filter.box_score_threshold: 0.01
        sv_dino_filter.box_isect_threshold: 0.1

        sv_depth_score.enabled: 0
        sv_depth_score.model_fpath:
            - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
        sv_depth_filter.threshold:
            - 0.10

        ##########################
        ## Cluster Sites Params ##
        ##########################
        cluster_sites.context_factor: 1.5
        cluster_sites.minimum_size: '128x128@2GSD'
        cluster_sites.maximum_size: '1024x1024@2GSD'

        ########################
        ## AC/SC CROP PARAMS  ##
        ########################
        sc_crop.target_gsd: 2GSD
        sc_crop.minimum_size: '128x128@2GSD'
        sc_crop.force_min_gsd: 2GSD
        sc_crop.context_factor: 1.0
        sc_crop.rpc_align_method: affine_warp
        sc_crop.sensor_to_time_window:
            - 'S2: 2month'

        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3
        sc_pxl.input_space_scale: 2GSD
        sc_pxl.window_space_scale: 2GSD
        sc_pxl.output_space_scale: 2GSD
        sc_pxl.chip_dims: '128,128'
        #sc_pxl.time_span: 6m
        #sc_pxl.time_sampling: auto
        #sc_pxl.time_steps: 12
        #sc_pxl.chip_dims: auto
        sc_pxl.set_cover_algo: null
        sc_pxl.resample_invalid_frames: 3
        sc_pxl.observable_threshold: 0.0
        sc_pxl.mask_low_quality: false
        sc_pxl.drop_unused_frames: true
        #sc_pxl.num_workers: 12
        #sc_pxl.batch_size: 1
        sc_pxl.write_workers: 0

        ########################
        ## AC/SC POLY PARAMS  ##
        ########################

        sc_poly.thresh:
            - 0.07
            - 0.10
            - 0.20
            - 0.30
            - 0.40
        sc_poly.boundaries_as: polys
        sc_poly.resolution: 8GSD
        sc_poly.min_area_square_meters: 7200

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_crop.enabled: 0
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 1
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_yardrat_for_scac_eval_and_dataset" \
    --devices="0," --tmux_workers=4 \
    --backend=tmux --queue_name "_yardrat_for_scac_eval_and_dataset" \
    --skip_existing=1 \
    --run=0


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_yardrat_for_scac_eval_and_dataset
    " \
    --output_dpath="$DVC_EXPT_DPATH/_yardrat_for_scac_eval_and_dataset/aggregate" \
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
        concise: 1
        show_csv: 0
    "
#ls "$DVC_EXPT_DPATH"/_yardrat_for_scac_eval_and_dataset


# We now need to run confusion analysis the BAS result for each region:
# We will use cmd-queue and some bash to run the jobs in parallel
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
cmd_queue new "confusion_queue"
for script in "$DVC_EXPT_DPATH"/_yardrat_for_scac_eval_and_dataset/eval/flat/bas_poly_eval/*/confusion_analysis.sh; do
    echo "script = $script"
    HASHID=$(echo "$script" | sha256sum | cut -c1-8)
    cmd_queue submit --jobname="$HASHID" -- confusion_queue \
        "$script" --viz_sites=False
done
cmd_queue show "confusion_queue"
cmd_queue run --workers=16 "confusion_queue" --backend=tmux


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)

# To prepare enriched annotations copy the original data into a new folder and
# then copy the hard negatives into the new folder (overwriting some of the old data)
cp -RLv "$DVC_DATA_DPATH"/annotations/drop7/. "$DVC_DATA_DPATH"/annotations/drop7-hard-v1

# Now overwrite the region / site models with the hard cases
for dpath in "$DVC_EXPT_DPATH"/_yardrat_for_scac_eval_and_dataset/eval/flat/bas_poly_eval/*/confusion_analysis/enriched_annots; do
    rsync -avprPRL "$dpath"/./region_models "$DVC_DATA_DPATH"/annotations/drop7-hard-v1
    rsync -avprPRL "$dpath"/./site_models "$DVC_DATA_DPATH"/annotations/drop7-hard-v1
done

# Add the new hard cases to DVC
sdvc add --verbose 3 -- "$DVC_DATA_DPATH"/annotations/drop7-hard-v1

(cd "$DVC_DATA_DPATH" && git commit -am "Update drop7 hard annots" && git push)

dvc push -r aws annotations/drop7-hard-v1 -vvv



#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch42_step3698.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch45_step3956.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch51_step4472.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch54_step4730.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch70_step6106.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/.gitignore
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch0_step86.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch1_step172.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch230_step19866.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch236_step20382.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch240_step20726.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch250_step21586.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch311_step26832.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/.gitignore
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch186_step16082.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/.gitignore
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch41_step3612.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch57_step19836.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/.gitignore
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch138_step11954.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch148_step12814.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch165_step14276.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch173_step14964.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch185_step15996.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch518_step22317.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch521_step22446.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/.gitignore
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch103_step8944.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch104_step9030.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch54_step4730.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch84_step7310.pt.dvc
#models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch95_step8256.pt.dvc
# models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch185_step15996.pt.dvc
# models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch311_step26832.pt.dvc
# models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch70_step6106.pt.dvc
# models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch186_step16082.pt.dvc


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DST_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')

TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop7-hard-v1

TRUTH_REGION_DPATH="$TRUTH_DPATH/region_models"
SRC_BUNDLE_DPATH=$DVC_DATA_DPATH/Aligned-Drop7
DST_BUNDLE_DPATH=$DST_DATA_DPATH/Drop7-Cropped2GSD-V2

test -d "$DVC_DATA_DPATH" || echo "ERROR: DVC_DATA_DPATH DOES NOT EXIST"
test -d "$DST_DATA_DPATH" || echo "ERROR: DST_DATA_DPATH DOES NOT EXIST"
test -d "$TRUTH_REGION_DPATH" || echo "ERROR: DVC_DATA_DPATH DOES NOT EXIST"
test -d "$SRC_BUNDLE_DPATH" || echo "ERROR: SRC_BUNDLE_DPATH DOES NOT EXIST"

# Find the region ids with annotations and data
export TRUTH_REGION_DPATH
export SRC_BUNDLE_DPATH
echo "TRUTH_REGION_DPATH = $TRUTH_REGION_DPATH"
echo "SRC_BUNDLE_DPATH = $SRC_BUNDLE_DPATH"



REGION_IDS_STR=$(python -c "if 1:
    import pathlib
    import os
    TRUTH_REGION_DPATH = os.environ.get('TRUTH_REGION_DPATH')
    SRC_BUNDLE_DPATH = os.environ.get('SRC_BUNDLE_DPATH')
    region_dpath = pathlib.Path(TRUTH_REGION_DPATH)
    src_bundle = pathlib.Path(SRC_BUNDLE_DPATH)
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

for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
done


### Cluster and Crop Jobs
python -m cmd_queue new "crop_for_sc_queue"
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    REGION_GEOJSON_FPATH=$TRUTH_REGION_DPATH/$REGION_ID.geojson
    REGION_CLUSTER_DPATH=$DST_BUNDLE_DPATH/$REGION_ID/clusters
    SRC_KWCOCO_FPATH=$SRC_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
    DST_KWCOCO_FPATH=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
    if ! test -f "$DST_KWCOCO_FPATH"; then
        cmd_queue submit --jobname="cluster-$REGION_ID" --depends="None" -- crop_for_sc_queue \
            python -m geowatch.cli.cluster_sites \
                --src "$REGION_GEOJSON_FPATH" \
                --minimum_size "256x256@2GSD" \
                --dst_dpath "$REGION_CLUSTER_DPATH" \
                --draw_clusters True

        python -m cmd_queue submit --jobname="crop-$REGION_ID" --depends="cluster-$REGION_ID" -- crop_for_sc_queue \
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
python -m cmd_queue show "crop_for_sc_queue"
python -m cmd_queue run --workers=8 "crop_for_sc_queue"


###
# Reproject Annotation Jobs
###



DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DST_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop7-hard-v1
TRUTH_REGION_DPATH="$TRUTH_DPATH/region_models"
BUNDLE_DPATH=$DST_DATA_DPATH/Drop7-Cropped2GSD-V2
# Find the region ids with annotations and data
export TRUTH_REGION_DPATH
export BUNDLE_DPATH
REGION_IDS_STR=$(python -c "if 1:
    import pathlib
    import os
    TRUTH_REGION_DPATH = os.environ.get('TRUTH_REGION_DPATH')
    BUNDLE_DPATH = os.environ.get('BUNDLE_DPATH')
    region_dpath = pathlib.Path(TRUTH_REGION_DPATH)
    bundle_dpath = pathlib.Path(BUNDLE_DPATH)
    region_fpaths = list(region_dpath.glob('*_[RC]*.geojson'))
    region_names = [p.stem for p in region_fpaths]
    final_names = []
    for region_name in region_names:
        src_coco_fpath = bundle_dpath / region_name / f'imgonly-{region_name}-rawbands.kwcoco.zip'
        dst_coco_fpath = bundle_dpath / region_name / f'imganns-{region_name}-rawbands.kwcoco.zip'
        if src_coco_fpath.exists():
            if not dst_coco_fpath.exists():
                final_names.append(region_name)
    print(' '.join(sorted(final_names)))
    ")
echo "REGION_IDS_STR = $REGION_IDS_STR"

# shellcheck disable=SC2206
REGION_IDS_ARR=($REGION_IDS_STR)
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
done

python -m cmd_queue new "reproject_for_sc"
# shellcheck disable=SC3054
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
    python -m cmd_queue submit --jobname="reproject-$REGION_ID" -- reproject_for_sc \
        geowatch reproject_annotations \
            --src "$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip" \
            --dst "$DST_BUNDLE_DPATH/$REGION_ID/imganns-$REGION_ID-rawbands.kwcoco.zip" \
            --io_workers="avail/2" \
            --region_models="$TRUTH_DPATH/region_models/${REGION_ID}.geojson" \
            --site_models="$TRUTH_DPATH/site_models/${REGION_ID}_*.geojson"
done
python -m cmd_queue show "reproject_for_sc"
python -m cmd_queue run --workers=8 "reproject_for_sc"


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m geowatch.cli.prepare_splits \
    --src_kwcocos "$DST_BUNDLE_DPATH"/*/imganns*-rawbands.kwcoco.zip \
    --dst_dpath "$DST_BUNDLE_DPATH" \
    --suffix=rawbands \
    --backend=tmux --tmux_workers=6 \
    --splits split6 \
    --run=1


DST_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DST_BUNDLE_DPATH=$DST_DATA_DPATH/Drop7-Cropped2GSD-V2

cd "$DST_BUNDLE_DPATH"

ls -- */*/subdata.kwcoco.*
rm -rf -- */*/subdata.kwcoco.*

dvc add -vvv -- */clusters

dvc add -vvv -- \
    *_rawbands_*.kwcoco.zip \
    */imgonly-*-rawbands.kwcoco.zip \
    */imganns-*-rawbands.kwcoco.zip \
    */*/L8 \
    */*/S2 \
    */*/WV \
    */*/PD \
    */*/WV1 && \
git commit -m "Update Drop7-Cropped2GSD-V2" && \
git push && \
dvc push -r aws -R . -vvv



##### FIXUP NAMES
__doc__="
Prepare teamfeat included the -rawbands suffix when applying feature names.
The 0.10.2 branch patches this, and this script fixes the existing names
"

DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware="auto")
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-Cropped2GSD-V2
cd "$BUNDLE_DPATH"
python -c "

import ubelt as ub
bundle_dpath = ub.Path('.').resolve()
bad_fpaths = list(bundle_dpath.glob('*/*rawbands*_*.kwcoco.zip'))

tracked_paths = []
untracked_paths = []

for fpath in bad_fpaths:
    dvc_fpath = fpath.augment(tail='.dvc')
    if dvc_fpath.exists():
        tracked_paths.append(fpath)
    else:
        untracked_paths.append(fpath)


mv_jobs = []
for fpath in untracked_paths:
    assert '-rawbands' in fpath.name
    new_fpath = fpath.parent / fpath.name.replace('-rawbands', '')
    mv_jobs.append((fpath, new_fpath))

for src, dst in ub.ProgIter(mv_jobs):
    src.move(dst)
"



#### Add in teamfeatures

# AC/SC Features on Clusters
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware="auto")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-Cropped2GSD-V2
python -m geowatch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/*/imgonly-*-rawbands.kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_invariants2=0 \
    --with_sam=0 \
    --with_materials=0 \
    --with_depth=1 \
    --with_mae=0 \
    --with_cold=0 \
    --skip_existing=0 \
    --assets_dname=teamfeats \
    --gres=0, \
    --cold_workermode=process \
    --cold_workers=8 \
    --tmux_workers=2 \
    --backend=tmux --run=0


# Reproject onto feature kwcocos
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DST_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop7-hard-v1
TRUTH_REGION_DPATH="$TRUTH_DPATH/region_models"
BUNDLE_DPATH=$DST_DATA_DPATH/Drop7-Cropped2GSD-V2
# Find the region ids with annotations and data
export TRUTH_REGION_DPATH
export BUNDLE_DPATH
REGION_IDS_STR=$(python -c "if 1:
    import pathlib
    import os
    TRUTH_REGION_DPATH = os.environ.get('TRUTH_REGION_DPATH')
    BUNDLE_DPATH = os.environ.get('BUNDLE_DPATH')
    region_dpath = pathlib.Path(TRUTH_REGION_DPATH)
    bundle_dpath = pathlib.Path(BUNDLE_DPATH)
    region_fpaths = list(region_dpath.glob('*_[RC]*.geojson'))
    region_names = [p.stem for p in region_fpaths]
    final_names = []
    for region_name in region_names:
        src_coco_fpath = bundle_dpath / region_name / f'combo_imgonly-{region_name}_D.kwcoco.zip'
        dst_coco_fpath = bundle_dpath / region_name / f'combo_imganns-{region_name}_D.kwcoco.zip'
        if src_coco_fpath.exists():
            if 0 or not dst_coco_fpath.exists():
                final_names.append(region_name)
    print(' '.join(sorted(final_names)))
    ")
echo "REGION_IDS_STR = $REGION_IDS_STR"

# shellcheck disable=SC2206
REGION_IDS_ARR=($REGION_IDS_STR)
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
done
python -m cmd_queue new "reproject_for_feat_sc"
# shellcheck disable=SC3054
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
    python -m cmd_queue submit --jobname="reproject-$REGION_ID" -- reproject_for_feat_sc \
        geowatch reproject_annotations \
            --src "$BUNDLE_DPATH/$REGION_ID/combo_imgonly-${REGION_ID}_D.kwcoco.zip" \
            --dst "$BUNDLE_DPATH/$REGION_ID/combo_imganns-${REGION_ID}_D.kwcoco.zip" \
            --io_workers="avail/2" \
            --region_models="$TRUTH_DPATH/region_models/${REGION_ID}.geojson" \
            --site_models="$TRUTH_DPATH/site_models/${REGION_ID}_*.geojson"
done
python -m cmd_queue show "reproject_for_feat_sc"
python -m cmd_queue run --workers=16 "reproject_for_feat_sc"


DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware="auto")
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-Cropped2GSD-V2
python -m geowatch.cli.prepare_splits \
    --src_kwcocos "$BUNDLE_DPATH"/*/combo_imganns-*_D.kwcoco.zip \
    --dst_dpath "$BUNDLE_DPATH" \
    --suffix=D \
    --backend=tmux --tmux_workers=6 \
    --splits split6 \
    --run=1

# Add team features in their respective output directories
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware="auto")
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-Cropped2GSD-V2
cd "$BUNDLE_DPATH"
ls -- */_assets/dzyne_depth
dvc add -vvv -- */_assets/dzyne_depth
#### TODO: fix the output feature names
dvc add -vvv -- */imganns-*-rawbands_dzyne_depth.kwcoco.zip */combo_imganns-*-rawbands_D.kwcoco.zip
dvc add -vvv -- data_train_D_split6.kwcoco.zip data_vali_D_split6.kwcoco.zip



DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop7-hard-v1

TRAIN_FPATH=/home/a.dhakal/active/proj_smart/smart_dvc/smart_drop7/Drop7-Cropped2GSD-Features/invariant_splits/data_train_I2_split6.kwcoco.zip
VALI_FPATH=/home/a.dhakal/active/proj_smart/smart_dvc/smart_drop7/Drop7-Cropped2GSD-Features/invariant_splits/data_vali_I2_split6.kwcoco.zip
geowatch reproject_annotations \
    --src "$TRAIN_FPATH" \
    --inplace=True \
    --io_workers="avail/2" \
    --region_models="$TRUTH_DPATH/region_models/*.geojson" \
    --site_models="$TRUTH_DPATH/site_models/*.geojson"

geowatch reproject_annotations \
    --src "$VALI_FPATH" \
    --inplace=True \
    --io_workers="avail/2" \
    --region_models="$TRUTH_DPATH/region_models/*.geojson" \
    --site_models="$TRUTH_DPATH/site_models/*.geojson"

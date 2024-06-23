Baseline 2023-10-12 Full
------------------------

This is the full BAS+SV+AC pipeline that extends the joint_bas_sc pipeline.

Currrently WIP, might not be totally right yet. Help wanted.


.. code:: bash

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)

    # Should contain the high resolution data needed for SC
    DVC_HIRES_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)

    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)

    geowatch schedule --params="
        pipeline: full

        # Convinience argument which uses SMART-specific assumptions
        # to correctly set the paths that the AC/SC clusters will be cropped
        # from.
        smart_highres_bundle: $DVC_HIRES_DATA_DPATH/Aligned-Drop7

        matrix:

            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt

            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt

            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/KR_R001/imgonly-KR_R001-rawbands.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/KR_R002/imgonly-KR_R002-rawbands.kwcoco.zip
                # Uncomment to run on more regions
                # - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/KW_C001/imgonly-KW_C001-rawbands.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CO_C001/imgonly-CO_C001-rawbands.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CN_C000/imgonly-CN_C000-rawbands.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/NZ_R001/imgonly-NZ_R001-rawbands.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/CH_R001/imgonly-CH_R001-rawbands.kwcoco.zip

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

            ######################
            ## SV Params Params ##
            ######################
            sv_crop.enabled: 1
            sv_crop.minimum_size: '256x256@2GSD'
            sv_crop.num_start_frames: 3
            sv_crop.num_end_frames: 3
            sv_crop.context_factor: 1.6

            sv_dino_boxes.enabled: 1
            sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
            sv_dino_boxes.window_dims: 256
            sv_dino_boxes.window_overlap: 0.5
            sv_dino_boxes.fixed_resolution: 3GSD

            sv_dino_filter.enabled: 1
            sv_dino_filter.end_min_score:
                - 0.15
            sv_dino_filter.start_max_score: 1.0
            sv_dino_filter.box_score_threshold: 0.01
            sv_dino_filter.box_isect_threshold: 0.1

            sv_depth_score.enabled: 1
            sv_depth_score.model_fpath:
                - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
            sv_depth_filter.threshold:
                - 0.10

            ##########################
            ## Cluster Sites Params ##
            ##########################
            cluster_sites.context_factor: 1.5
            cluster_sites.minimum_size: '128x128@8GSD'
            cluster_sites.maximum_size: '1024x1024@8GSD'

            ########################
            ## AC/SC CROP PARAMS  ##
            ########################
            sc_crop.target_gsd: 8GSD
            sc_crop.minimum_size: '128x128@8GSD'
            sc_crop.force_min_gsd: 8GSD
            sc_crop.context_factor: 1.0
            sc_crop.rpc_align_method: affine_warp
            sc_crop.sensor_to_time_window:
                - 'S2: 1month'

            ########################
            ## AC/SC PIXEL PARAMS ##
            ########################

            sc_pxl.tta_fliprot: 0.0
            sc_pxl.tta_time: 0.0
            sc_pxl.chip_overlap: 0.3
            sc_pxl.input_space_scale: 8GSD
            sc_pxl.window_space_scale: 8GSD
            sc_pxl.output_space_scale: 8GSD
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
            sc_poly.boundaries_as:
                - polys
                - bounds
            sc_poly.resolution: 8GSD
            sc_poly.min_area_square_meters: 7200
            sc_poly.new_algo: crall

            #############################
            ## AC/SC POLY EVAL PARAMS  ##
            #############################

            sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
            sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models

            ##################################
            ## HIGH LEVEL PIPELINE CONTROLS ##
            ##################################
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl.enabled: 1
            bas_pxl_eval.enabled: 1
            sc_crop.enabled: 1
            sc_poly.enabled: 1
            sc_poly_eval.enabled: 1
            sc_pxl.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly_viz.enabled: 0
            bas_poly_viz.enabled: 0
        " \
        --root_dpath="$DVC_EXPT_DPATH/_baseline_2023-10-12_full_pipeline" \
        --devices="0," --tmux_workers=4 \
        --backend=tmux --queue_name "_baseline_2023-10-12_full_pipeline" \
        --skip_existing=1 \
        --run=0


Baseline 2023-06-22 Joint AC/SC Truth
-------------------------------------

The following is an SC baseline that uses the ground truth to pre-crop regions. Please see the docs in baseline_joint_bas_sc_2023-06-22.rst for more comments on the general process.



Use this code to get a feel for what parameters are available / required

.. code:: python

    from watch.mlops.smart_pipeline import *  # NOQA
    dag = make_smart_pipeline('crop_sc')

    # Show the graph structure of inputs and outputs
    dag.print_graphs()

    # List what known parameters are configurable
    dag.inspect_configurables()




.. code:: bash

    HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

    python -m geowatch.mlops.schedule_evaluation --params="
        matrix:

            ########################
            ## SC CROPPING PARAMS ##
            ########################

            sc_crop.crop_src_fpath:
              #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/KR_R001/imgonly-KR_R001.kwcoco.zip
              - $HIRES_DVC_DATA_DPATH/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip
              #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/AE_R001/imgonly-AE_R001.kwcoco.zip
              #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/BR_R002/imgonly-BR_R002.kwcoco.zip
              #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/CH_R001/imgonly-CH_R001.kwcoco.zip
              #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/NZ_R001/imgonly-NZ_R001.kwcoco.zip
              #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/PE_R001/imgonly-PE_R001.kwcoco.zip
              #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/BR_R004/imgonly-BR_R004.kwcoco.zip

            # Because there is no BAS component here, we have to hard-code what
            # the annotations to use are. We will just use the truth annotations
            # here, but you could swap these out with specific BAS preditions of
            # interest.  (You can also use submatrices to tie regions - i.e.
            # site summaries - to crop paths, but if the polygons dont overlap
            # the data they are ignored, so we can just specify a glob of
            # everything here.)
            sc_crop.regions: '$HIRES_DVC_DATA_DPATH/annotations/drop6/region_models/*.geojson'

            sc_crop.force_nodata: -9999
            sc_crop.include_channels: 'red|green|blue|quality'
            sc_crop.exclude_sensors: 'L8'
            sc_crop.minimum_size: '128x128@8GSD'
            sc_crop.convexify_regions: True
            sc_crop.target_gsd: 2
            sc_crop.context_factor: 1.5
            sc_crop.force_min_gsd: 8
            sc_crop.img_workers: 16
            sc_crop.aux_workers: 2

            #####################
            ## SC PIXEL PARAMS ##
            #####################

            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            sc_pxl.tta_fliprot: 0.0
            sc_pxl.tta_time: 0.0
            sc_pxl.chip_overlap: 0.3
            sc_pxl.input_space_scale: 8GSD
            sc_pxl.window_space_scale: 8GSD
            sc_pxl.output_space_scale: 8GSD
            sc_pxl.time_span: 6m
            sc_pxl.time_sampling: auto
            sc_pxl.time_steps: 12
            sc_pxl.chip_dims: auto
            sc_pxl.set_cover_algo: null
            sc_pxl.resample_invalid_frames: 3
            sc_pxl.observable_threshold: 0.0
            sc_pxl.mask_low_quality: true
            sc_pxl.drop_unused_frames: true
            sc_pxl.num_workers: 12
            sc_pxl.batch_size: 1
            sc_pxl.write_workers: 0

            #####################
            ## SC POLY PARAMS  ##
            #####################

            sc_poly.thresh: 0.07
            sc_poly.boundaries_as: polys
            sc_poly.resolution: 8GSD
            sc_poly.min_area_square_meters: 7200

            ##########################
            ## SC POLY EVAL PARAMS  ##
            ##########################

            sc_poly_eval.true_site_dpath: $HIRES_DVC_DATA_DPATH/annotations/drop6/site_models
            sc_poly_eval.true_region_dpath: $HIRES_DVC_DATA_DPATH/annotations/drop6/region_models

            ##################################
            ## HIGH LEVEL PIPELINE CONTROLS ##
            ##################################
            sc_crop.enabled: 1
            sc_pxl.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly.enabled: 1
            sc_poly_eval.enabled: 1
            sc_poly_viz.enabled: 0
        " \
        --pipeline=crop_sc \
        --root_dpath="$DVC_EXPT_DPATH/_drop7_baseline_sc_truth2" \
        --queue_name "_drop7_baseline_sc_truth" \
        --devices=",1" \
        --backend=tmux --tmux_workers=6 \
        --cache=1 --skip_existing=1 --run=1





The process level graph for this pipeline looks like this:


.. code::

    ╙── sc_crop
        ╽
        sc_pxl
        ├─╼ sc_pxl_eval
        └─╼ sc_poly
            ├─╼ sc_poly_eval
            └─╼ sc_poly_viz

Baseline 2023-06-22 Joint BAS+AC/SC
-----------------------------------

The following is a baseline joint BAS + AC/SC grid that runs BAS, crops the
candidates, and then runs the baseline AC/SC model. This is effectively a
superset of the corresponding BAS model for this date.


The most important difference is that the ``--pipeline=bas`` parameter changed
to  ``--pipeline=joint_bas_sc``, which provides a different "pipeline template"
that the YAML parameter grid will be used to configure.


To get a sense of what the parameters are that are necessary for the user to
specify, you can make an instance of one of these template pipelines and then
inspect the graphs structure as well as the "known configurable parameters".

.. code:: python

    from watch.mlops.smart_pipeline import *  # NOQA
    dag = make_smart_pipeline('joint_bas_sc')

    # Show the graph structure of inputs and outputs
    dag.print_graphs()

    # List what known parameters are configurable
    dag.inspect_configurables()

    # (Note: you can always specify
    # unknown parameters for any node by using
    # ``<node_name>.<param_name>: <list of options>``, which will result in the
    # bash call to the node associated with ``<node_name>`` to get an extra
    # ``--<param_name>=<val>`` in its bash invocation. Make use of the
    #``--print-commands`` options when running schedule. to see these.


In the above configurables you will see that there is a table of consiting of

* nodes: the name of the mlops node,
* keys: parameter names associated with the node
* connected: if it is automatically connected as in the pipeline template or not
* type: if the parmater is an "algo_param" (chances algorithm behavior), "perf_param" (changes algorithm resource usage, but not results), "in_path" (a required input), or "out_path" (a node output, which are always connected for you).
* maybe_required: a flag that indicates if the user (might) need to set this to work. Some of these are optional, but currently mlops cant distinguish this. In general, it's a good idea to specify these if possible.

The ones to pay extra attention to are the in_paths that might be required.
These are likely inputs to the algorithm that you have to specify where to get
them from.

.. code:: bash

    # Note the location of the low resolution and the high resolution data
    # might be different. (E.g. I have the lowres data on my SSD and the highres
    # data on my HDD)
    LORES_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)

    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

    python -m geowatch.mlops.schedule_evaluation --params="
        matrix:

            ######################
            ## BAS PIXEL PARAMS ##
            ######################

            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            bas_pxl.test_dataset:
                - $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
                #- $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
                #- $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
                #- $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
                #- $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
                #- $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
                #- $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
                #- $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims:
                - [196,196]
            bas_pxl.time_span: auto
            bas_pxl.input_space_scale: 10GSD
            bas_pxl.time_sampling: soft4

            ######################
            ## BAS POLY PARAMS  ##
            ######################

            bas_poly.thresh: 0.425
            bas_poly.time_thresh: 0.8
            bas_poly.inner_window_size: 1y
            bas_poly.inner_agg_fn: max
            bas_poly.norm_ord: inf
            bas_poly.moving_window_size: null
            bas_poly.poly_merge_method: 'v2'
            bas_poly.polygon_simplify_tolerance: 1
            bas_poly.agg_fn: probs
            bas_poly.min_area_square_meters: 7200
            bas_poly.max_area_square_meters: 8000000
            bas_poly.boundary_region: $LORES_DVC_DATA_DPATH/annotations/drop6/region_models

            ###########################
            ## BAS POLY EVAL PARAMS  ##
            ###########################

            bas_poly_eval.true_site_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/site_models
            bas_poly_eval.true_region_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/region_models

            ########################
            ## SC CROPPING PARAMS ##
            ########################

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

            sc_poly_eval.true_site_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/site_models
            sc_poly_eval.true_region_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/region_models

            ##################################
            ## HIGH LEVEL PIPELINE CONTROLS ##
            ##################################
            bas_pxl.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            sc_crop.enabled: 1
            sc_pxl.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly.enabled: 1
            sc_poly_eval.enabled: 1
            bas_poly_viz.enabled: 0
            sc_poly_viz.enabled: 0

        submatrices:
            - bas_pxl.input_space_scale: 10GSD
              bas_pxl.window_space_scale: 10GSD
              bas_pxl.output_space_scale: 10GSD
              bas_poly.resolution: 10GSD

        submatrices1:
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/KR_R001/imgonly-KR_R001.kwcoco.zip
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/AE_R001/imgonly-AE_R001.kwcoco.zip
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/BR_R002/imgonly-BR_R002.kwcoco.zip
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/CH_R001/imgonly-CH_R001.kwcoco.zip
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/NZ_R001/imgonly-NZ_R001.kwcoco.zip
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/PE_R001/imgonly-PE_R001.kwcoco.zip
            - bas_pxl.test_dataset: $LORES_DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
              sc_crop.crop_src_fpath: $HIRES_DVC_DATA_DPATH/Aligned-Drop7/BR_R004/imgonly-BR_R004.kwcoco.zip
        " \
        --pipeline=joint_bas_sc \
        --root_dpath="$DVC_EXPT_DPATH/_drop7_nowinter_baseline_joint_bas_sc" \
        --queue_name "_drop7_nowinter_baseline_joint_bas_sc" \
        --devices="0,1" \
        --backend=tmux --tmux_workers=6 \
        --cache=1 --skip_existing=1 --run=1


The above submatrices "tie" high res dataset to low res dataset needed by the
cropping step. These are needed because the BAS algorithm starts working on the
lowres dataset, but eventually requires information from the highres data when
it gets to the sc crop step. I used the following code can help generate these
submatrices.

.. code:: bash

    ### Helper to build SV crop dataset submatrix
    python -c "if 1:
        import ubelt as ub
        regions = ['KR_R001', 'KR_R002', 'AE_R001', 'BR_R002', 'CH_R001', 'NZ_R001', 'PE_R001', 'BR_R004']
        feature_code = 'EI2LMSC'
        dollar = chr(36)
        dvc_var1 = dollar + 'LORES_DVC_DATA_DPATH'
        dvc_var2 = dollar + 'HIRES_DVC_DATA_DPATH'
        for region_id in regions:
            print(ub.codeblock(
                f'''
                - bas_pxl.test_dataset: {dvc_var1}/Drop7-MedianNoWinter10GSD/combo_imganns-{region_id}_{feature_code}.kwcoco.zip
                  sc_crop.crop_src_fpath: {dvc_var2}/Aligned-Drop7/{region_id}/imgonly-{region_id}.kwcoco.zip
                '''))
    "


The process graph for this pipeline look like this:

.. code:: bash

    Process Graph
    ╙── bas_pxl
        ├─╼ bas_pxl_eval
        └─╼ bas_poly
            ├─╼ sc_crop
            │   ╽
            │   sc_pxl
            │   ├─╼ sc_pxl_eval
            │   └─╼ sc_poly ╾ bas_poly
            │       ├─╼ sc_poly_eval
            │       └─╼ sc_poly_viz
            ├─╼ bas_poly_eval
            ├─╼ bas_poly_viz
            └─╼  ...

To report your scores:

.. code:: bash

    # Pull out baseline tables
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m geowatch.mlops.aggregate \
        --pipeline=joint_bas_sc \
        --target "
            - $DVC_EXPT_DPATH/_drop7_nowinter_baseline_joint_bas_sc
        " \
        --output_dpath="$DVC_EXPT_DPATH/_drop7_nowinter_baseline_joint_bas_sc/aggregate" \
        --resource_report=0 \
        --eval_nodes="
            - sc_poly_eval
            #- bas_poly_eval
            #- bas_pxl_eval
        " \
        --plot_params="
            enabled: 0
            stats_ranking: 0
            min_variations: 1
        " \
        --stdout_report="
            top_k: 10
            per_group: 1
            macro_analysis: 0
            analyze: 0
            print_models: True
            reference_region: final
        " \
        --rois="auto"


Note: in the current version there seems to be some sort of bug and this is
producing zero SC F1 scores.



Tips and Tricks
---------------

To get a better senese of exactly what the pipeline is doing set ``--run=0``,
``--skip_existing=0``, add the ``--print-commands`` argument, set
``--backend=serial``, ``--cache=False`` and comment out all execpt one of the
``bas_pxl.test_dataset`` entries. This will print a list of the exact bash
commands that the pipeline will run.

Because there is only one input region, the sequence of commands would be
exactly what you would execute to run to manually execute the pipeline.

For this joint bas + sc case, you will see the following sequence:

* a BAS fusion predict step on the bas pixel test dataset using your specified package and params
* a bas pixel evaluation step
* a run tracker step to turn the bas pixel heatmaps into polygons
* a run metrics framework step that evaluates the bas polygon predictions
* a coco-align step that crops the high res data using the polygons output by bas-poly
* a SC fusion predict step that is run on the output of the cropped high res dataset
* a pixel evaluation on the SC pixel predictions
* a tracker step to convert the SC heatmaps to polygons
* a run metrics step to evaluate the SC polygons

You will also set a "network text" graph that shows the dependencies between
these steps.

Note: the exact order might shift as long as all dependencies needed by a step have been met.


Troubleshooting
---------------

The most basic way to debug a failure is to switch to serial mode, but there
are also efficient ways to do this with the tmux backend.

When a tmux pipeline fails, there are several ways you can debug. You can
``tmux a`` to attach to an existing tmux sessions and then ``<ctrl-b>``
followed by ``s`` to view all sessions interactively. Navigate to the failed
session and look at the logs.

If a run failed and you just want to get rid of all of the cmd-queue tmux sessions use the cmd-queue CLI as such:


.. code::  bash

   cmd_queue cleanup

which will exit all the tmux sessions cmd_queu started.

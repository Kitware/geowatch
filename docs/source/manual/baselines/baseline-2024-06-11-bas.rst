Baseline 2024-06-11 BAS
-----------------------

The following is the EVAL23 baseline MLOPs grid for BAS-only.


.. code:: python

    from geowatch.mlops.smart_pipeline import *  # NOQA
    dag = make_smart_pipeline('bas')

    # Show the graph structure of inputs and outputs
    dag.print_graphs()

    # List what known parameters are configurable
    dag.inspect_configurables()


.. code:: bash

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
    TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop8
    MLOPS_NAME=_bas_only_baseline
    MLOPS_DPATH=$DVC_EXPT_DPATH/$MLOPS_NAME
    # Set this to the GPU numbers you want to use.
    DEVICES="1,2"

    MODEL_SHORTLIST="
    - $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch187_step2632.pt
    - $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
    "

    mkdir -p "$MLOPS_DPATH"
    echo "$MODEL_SHORTLIST" > "$MLOPS_DPATH/shortlist.yaml"

    cat "$MLOPS_DPATH/shortlist.yaml"

    geowatch schedule --params="
        pipeline: bas

        matrix:
            bas_pxl.package_fpath: $MLOPS_DPATH/shortlist.yaml

            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip
                - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/CN_C000/imganns-CN_C000-rawbands.kwcoco.zip
                - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/KW_C001/imganns-KW_C001-rawbands.kwcoco.zip
                - $DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1/CO_C001/imganns-CO_C001-rawbands.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims: auto
            bas_pxl.time_span: auto
            bas_pxl.time_sampling: soft4
            bas_poly.thresh:
                #- 0.10
                #- 0.30
                - 0.35
                #- 0.4
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
            bas_poly.time_pad_after: 3 months
            bas_poly.time_pad_before: 3 months
            bas_poly.min_area_square_meters: 7200
            bas_poly.max_area_square_meters: 8000000
            bas_poly.boundary_region: $TRUTH_DPATH/region_models
            bas_poly_eval.true_site_dpath: $TRUTH_DPATH/site_models
            bas_poly_eval.true_region_dpath: $TRUTH_DPATH/region_models
            bas_pxl.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 0
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
        " \
        --root_dpath="$MLOPS_DPATH" \
        --devices="$DEVICES" --tmux_workers=4 \
        --backend=tmux --queue_name "$MLOPS_NAME" \
        --skip_existing=1 \
        --run=1


The process graph for this pipeline looks like:


.. code::

    Process Graph
    ╙── bas_pxl
        ├─╼ bas_pxl_eval
        └─╼ bas_poly
            ├─╼ bas_poly_eval
            └─╼ bas_poly_viz


To report your scores:

.. code:: bash

    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
    MLOPS_DPATH=$DVC_EXPT_DPATH/_bas_only_baseline
    echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"

    python -m geowatch.mlops.aggregate \
        --pipeline=bas \
        --target "
            - $MLOPS_DPATH
        " \
        --export_tables=0 \
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
            #params_of_interest:
            #    - params.bas_poly.thresh
            #    - resolved_params.bas_pxl.channels
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
        --rois="KR_R002,CN_C000,KW_C001,CO_C001"

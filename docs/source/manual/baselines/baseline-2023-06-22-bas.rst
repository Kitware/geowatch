Baseline 2023-06-22 BAS
-----------------------

The following is the EVAL11 baseline MLOPs grid.



.. code:: python

    from watch.mlops.smart_pipeline import *  # NOQA
    dag = make_smart_pipeline('bas')

    # Show the graph structure of inputs and outputs
    dag.print_graphs()

    # List what known parameters are configurable
    dag.inspect_configurables()


.. code:: bash

    # Eval11 Baseline on Drop7-MedianNoWinter10GSD
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
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
                - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
                - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
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
                - 0.425
            bas_poly.time_thresh:
                - 0.8
            bas_poly.inner_window_size:
                - 1y
            bas_poly.inner_agg_fn:
                ### NOTE: this should have been mean!
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

    # Pull out baseline tables
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m geowatch.mlops.aggregate \
        --pipeline=joint_bas_sc \
        --target "
            - $DVC_EXPT_DPATH/_drop7_nowinter_baseline
        " \
        --output_dpath="$DVC_EXPT_DPATH/_drop7_nowinter_baseline/aggregate" \
        --resource_report=0 \
        --eval_nodes="
            - bas_poly_eval
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


This will result in something like this table:

.. code::

               region_id  param_hashid  bas_faa_f1  bas_tp  bas_fp  bas_fn   bas_tpr    bas_f1  bas_ffpa
    7            AE_R001  fovtyjydzdjx    0.451815   183.0   279.0    76.0  0.706600  0.507600  0.109900
    0            BR_R002  fovtyjydzdjx    0.397880     2.0     5.0     1.0  0.666700  0.400000  0.005300
    5            BR_R004  fovtyjydzdjx    0.220783     6.0    39.0     1.0  0.857100  0.230800  0.043400
    4            CH_R001  fovtyjydzdjx    0.421669    37.0    87.0    12.0  0.755100  0.427700  0.014100
    1            KR_R001  fovtyjydzdjx    0.688673     8.0     6.0     1.0  0.888900  0.695700  0.010100
    2            KR_R002  fovtyjydzdjx    0.572208    17.0    11.0    14.0  0.548400  0.576300  0.007100
    6            NZ_R001  fovtyjydzdjx    0.486650    15.0    27.0     3.0  0.833300  0.500000  0.026700
    3            PE_R001  fovtyjydzdjx    0.055333     1.0    31.0     3.0  0.250000  0.055600  0.004800
    0    macro_08_0bcb55  fovtyjydzdjx    0.411877   269.0   485.0   111.0  0.688262  0.424212  0.027675

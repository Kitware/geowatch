from watch.mlops.aggregate import AggregateEvluationConfig, build_tables, build_aggregators
import ubelt as ub
import watch
data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
cmdline = 0
kwargs = {
    'root_dpath': expt_dvc_dpath / '_testpipe',
    'pipeline': 'bas',
    'io_workers': 2,
    # 'pipeline': 'joint_bas_sc_nocrop',
    # 'root_dpath': expt_dvc_dpath / '_testsc',
    #'pipeline': 'sc',
}

config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)
eval_type_to_results = build_tables(config)
eval_type_to_aggregator = build_aggregators(eval_type_to_results)
agg = eval_type_to_aggregator['bas_poly_eval']


model_name = 'Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704'
agg1 = agg.filterto(models=model_name)
agg1.build_macro_tables()
agg1.report_best()


mlops_cmd = ub.codeblock(
    r'''
    #!/bin/bash
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m watch.mlops.schedule_evaluation \
        --params="
            matrix:
                bas_pxl.package_fpath:
                    - $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_0/checkpoints/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
                bas_pxl.test_dataset:
                    # - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.json
                    - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.json
                bas_pxl.chip_overlap: 0.3
                bas_pxl.chip_dims:
                    # - auto
                    - 256,256
                bas_pxl.time_span: auto
                bas_pxl.time_sampling: auto
                bas_poly.thresh:
                    - 0.17
                bas_poly.polygon_simplify_tolerance:
                    - 1
                bas_poly.moving_window_size:
                    - null
                #bas_poly.min_area_sqkm:
                #    - 0.072
                #    #- 0.031
                #    - 0.001
                bas_poly.max_area_sqkm:
                    - null
                bas_pxl.enabled: 1
                bas_poly.enabled: 1
                bas_poly_eval.enabled: 1
                bas_pxl_eval.enabled: 0
                bas_poly_viz.enabled: 0
                sitecrop.crop_src_fpath:
                    - $DVC_DATA_DPATH/Drop4-SC/data_vali.kwcoco.json
                sc_pxl.package_fpath:
                    - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
        " \
        --root_dpath="$DVC_EXPT_DPATH/_testpipe" \
        --devices="1" --queue_size=1 \
        --backend=tmux --queue_name "endtoend-queue3" \
        --pipeline=joint_bas_sc --skip_existing=0 \
        --run=1
    ''')

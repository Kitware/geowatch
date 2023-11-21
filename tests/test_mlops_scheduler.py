"""
Checks that the scheduler builds appropriate commands.
"""


def test_bas_pipline_schedule_default_params():
    from geowatch.mlops import schedule_evaluation
    import ubelt as ub
    dpath = ub.Path.appdir('geowatch/unit_tests/scheduler/unit0').ensuredir()

    dvc_data_dpath = ub.Path('/my_data_dvc')

    config = schedule_evaluation.ScheduleEvaluationConfig(**{
        'run': 0,
        'root_dpath': dpath,
        'pipeline': 'bas',
        'backend': 'serial',
        'enable_links': False,
        'params': ub.codeblock(
            f'''
            bas_poly_eval.true_site_dpath: {dvc_data_dpath}/annotations/site_models
            bas_poly_eval.true_region_dpath: {dvc_data_dpath}/annotations/region_models
            '''
        )
    })
    dag, queue = schedule_evaluation.schedule_evaluation(config)

    bas_poly_job = None
    bas_poly_eval_job = None
    for job in queue.jobs:
        if job.name.startswith('bas_poly_id'):
            bas_poly_job = job
        if job.name.startswith('bas_poly_eval_id'):
            bas_poly_eval_job = job

    assert bas_poly_job is not None
    assert bas_poly_eval_job is not None
    assert "--boundary_region=None" in bas_poly_job.command


def test_bas_pipline_schedule1():
    from geowatch.mlops import schedule_evaluation
    import ubelt as ub
    dpath = ub.Path.appdir('geowatch/unit_tests/scheduler/unit1').ensuredir()

    dvc_data_dpath = ub.Path('/my_data_dvc')
    dvc_expt_dpath = ub.Path('/my_expt_dvc')

    config = schedule_evaluation.ScheduleEvaluationConfig(**{
        'run': 0,
        'root_dpath': dpath,
        'pipeline': 'bas',
        'backend': 'serial',
        'enable_links': False,
        'params': ub.codeblock(
            f'''
            bas_pxl.package_fpath:
                - {dvc_expt_dpath}/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            bas_pxl.test_dataset:
                - {dvc_data_dpath}/Drop6_MeanYear/imganns-KR_R001.kwcoco.zip
            bas_poly.boundary_region: {dvc_data_dpath}/annotations/region_models
            bas_poly_eval.true_site_dpath: {dvc_data_dpath}/annotations/site_models
            bas_poly_eval.true_region_dpath: {dvc_data_dpath}/annotations/region_models
            '''
        )
    })
    dag, queue = schedule_evaluation.schedule_evaluation(config)

    bas_poly_job = None
    bas_poly_eval_job = None
    for job in queue.jobs:
        if job.name.startswith('bas_poly_id'):
            bas_poly_job = job
        if job.name.startswith('bas_poly_eval_id'):
            bas_poly_eval_job = job

    assert bas_poly_job is not None
    assert bas_poly_eval_job is not None
    assert "--boundary_region 'None'" not in bas_poly_job.command


def test_joint_bas_sc_pipline_schedule1():
    from geowatch.mlops import schedule_evaluation
    import ubelt as ub
    dpath = ub.Path.appdir('geowatch/unit_tests/scheduler/unit2').ensuredir()

    dvc_data_dpath = ub.Path('/my_data_dvc')
    dvc_expt_dpath = ub.Path('/my_expt_dvc')

    config = schedule_evaluation.ScheduleEvaluationConfig(**{
        'run': 0,
        'root_dpath': dpath,
        'pipeline': 'joint_bas_sc',
        'backend': 'serial',
        'enable_links': False,
        'params': ub.codeblock(
            f'''
            bas_pxl.package_fpath:
                - {dvc_expt_dpath}/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            bas_pxl.test_dataset:
                - {dvc_data_dpath}/Drop6_MeanYear/imganns-KR_R001.kwcoco.zip
            bas_poly.boundary_region: {dvc_data_dpath}/annotations/region_models
            bas_poly_eval.true_site_dpath: {dvc_data_dpath}/annotations/site_models
            bas_poly_eval.true_region_dpath: {dvc_data_dpath}/annotations/region_models
            sc_poly_eval.true_site_dpath: {dvc_data_dpath}/annotations/site_models
            sc_poly_eval.true_region_dpath: {dvc_data_dpath}/annotations/region_models
            '''
        )
    })
    dag, queue = schedule_evaluation.schedule_evaluation(config)

    bas_poly_job = None
    bas_poly_eval_job = None
    for job in queue.jobs:
        if job.name.startswith('bas_poly_id'):
            bas_poly_job = job
        if job.name.startswith('bas_poly_eval_id'):
            bas_poly_eval_job = job

    assert bas_poly_job is not None
    assert bas_poly_eval_job is not None
    assert "--boundary_region 'None'" not in bas_poly_job.command

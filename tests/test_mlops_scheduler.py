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


def demodata_pipeline(dpath):
    import ubelt as ub
    script_fpath = dpath / 'script.py'
    pipeline_fpath = dpath / '_simple_demo_pipeline_v003.py'

    script_text = ub.codeblock(
        '''
        #!/usr/bin/env python3
        import scriptconfig as scfg
        import ubelt as ub
        import json


        class ScriptCLI(scfg.DataConfig):
            src = 'input.json'
            dst = 'output.json'
            param1 = None
            param2 = None
            param3 = None

            @classmethod
            def main(cls, cmdline=1, **kwargs):
                import rich
                from rich.markup import escape
                config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
                rich.print('config = ' + escape(ub.urepr(config, nl=1)))
                src_fpath = ub.Path(config.src)
                dst_fpath = ub.Path(config.dst)
                src_text = src_fpath.read_text()
                src_data = json.loads(src_text)

                hidden = int(ub.hash_data([config.param1, config.param2, config.param3], base=10, hasher='sha1'))
                flags = [c == '1' for c in bin(hidden)[2:]]
                goodness = sum(flags) / len(flags)

                dst_data = {'size': len(src_text), 'goodness': goodness, 'nest': src_data}
                dst_fpath.parent.ensuredir()
                dst_fpath.write_text(json.dumps(dst_data))

        __cli__ = ScriptCLI

        if __name__ == '__main__':
            __cli__.main()
        ''')
    # Test the code compiles and write it to disk
    compile(script_text, mode='exec', filename='<test-compile>')
    script_fpath.write_text(script_text)

    pipeline_text = ub.codeblock(
        '''
        from geowatch.mlops.pipeline_nodes import ProcessNode
        from geowatch.mlops.pipeline_nodes import PipelineDAG

        class Step1(ProcessNode):
            name = 'step1'
            executable = 'python ''' + str(script_fpath) + ''''
            in_paths = {
                'src',
            }
            out_paths = {
                'dst': 'step1_output.json',
            }

            def load_result(self, node_dpath):
                import json
                from geowatch.mlops.aggregate_loader import new_process_context_parser
                from geowatch.utils import util_dotdict
                fpath = node_dpath / self.out_paths[self.primary_out_key]
                data = json.loads(fpath.read_text())
                nest_resolved = {}
                nest_resolved['metrics.size'] = data['size']
                nest_resolved['metrics.goodness'] = data['goodness']
                flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
                flat_resolved = flat_resolved.insert_prefix(self.name, index=1)
                return flat_resolved

        def build_pipeline():
            nodes = {}
            nodes['step1'] = Step1()
            dag = PipelineDAG(nodes)
            dag.build_nx_graphs()
            return dag
        ''')

    # Test that the code compiles
    compile(pipeline_text, mode='exec', filename='<test-compile>')
    pipeline_fpath.write_text(pipeline_text)
    return pipeline_fpath


def test_simple_slurm_dry_run():
    """
    Ignore:
        python ~/code/geowatch/tests/test_mlops_scheduler.py test_simple_but_real_custom_pipeline

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/geowatch/tests'))
        from test_mlops_scheduler import *  # NOQA
    """
    from geowatch.mlops import schedule_evaluation
    import ubelt as ub
    dpath = ub.Path.appdir('geowatch/unit_tests/scheduler/test_slurm_dryrun').ensuredir()

    pipeline_fpath = demodata_pipeline(dpath)

    input_fpath = dpath / 'input.json'
    input_fpath.write_text('{"type": "orig_input"}')

    root_dpath = (dpath / 'runs').delete().ensuredir()
    config = schedule_evaluation.ScheduleEvaluationConfig(**{
        'run': 0,
        'root_dpath': root_dpath,
        'backend': 'slurm',
        'params': ub.codeblock(
            f'''
            pipeline: {pipeline_fpath}::build_pipeline()
            matrix:
                step1.src:
                    - {input_fpath}
                step1.param1: |
                    - this: "is text 100% representing"
                      some: "yaml config"
                      omg: "single ' quote"
                      eek: 'double " quote'
                step1.param2:
                    - option1
                    - option2
                step1.param3:
                    - 4.5
                    - 9.2
                    - 3.14159
                    - 2.71828
            '''
        )
    })

    print('Dry run first')
    config['run'] = 0
    dag, queue = schedule_evaluation.schedule_evaluation(config)


def test_simple_but_real_custom_pipeline():
    """
    Ignore:
        python ~/code/geowatch/tests/test_mlops_scheduler.py test_simple_but_real_custom_pipeline

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/geowatch/tests'))
        from test_mlops_scheduler import *  # NOQA
    """
    from geowatch.mlops import schedule_evaluation
    from geowatch.mlops import aggregate
    import ubelt as ub
    dpath = ub.Path.appdir('geowatch/unit_tests/scheduler/test_real_pipeline').ensuredir()

    pipeline_fpath = demodata_pipeline(dpath)

    input_fpath = dpath / 'input.json'
    input_fpath.write_text('{"type": "orig_input"}')

    root_dpath = (dpath / 'runs').delete().ensuredir()
    config = schedule_evaluation.ScheduleEvaluationConfig(**{
        'run': 0,
        'root_dpath': root_dpath,
        'backend': 'serial',
        'params': ub.codeblock(
            f'''
            pipeline: {pipeline_fpath}::build_pipeline()
            matrix:
                step1.src:
                    - {input_fpath}
                step1.param1: |
                    - this: "is text 100% representing"
                      some: "yaml config"
                      omg: "single ' quote"
                      eek: 'double " quote'
                step1.param2:
                    - option1
                    - option2
                step1.param3:
                    - 4.5
                    - 9.2
                    - 3.14159
                    - 2.71828
            '''
        )
    })

    print('Dry run first')
    config['run'] = 0
    dag, queue = schedule_evaluation.schedule_evaluation(config)

    print('Real run second')
    config['run'] = 1
    dag, queue = schedule_evaluation.schedule_evaluation(config)

    # Test that all job config files are readable
    import json
    for job_config_fpath in dag.root_dpath.glob('flat/step1/*/job_config.json'):
        config = json.loads(job_config_fpath.read_text())

    # Can we test that this is well formatted?
    for invoke_fpath in dag.root_dpath.glob('flat/step1/*/invoke.sh'):
        command = invoke_fpath.read_text()
        command

    agg_config = aggregate.AggregateEvluationConfig(
        target=root_dpath,
        pipeline=f'{pipeline_fpath}::build_pipeline()',
        output_dpath=(root_dpath / 'aggregate'),
        io_workers=0,
        eval_nodes=['step1'],
    )
    eval_type_to_aggregator = aggregate.run_aggregate(agg_config)
    agg = eval_type_to_aggregator['step1']
    print(f'agg = {ub.urepr(agg, nl=1)}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/geowatch/tests/test_mlops_scheduler.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)

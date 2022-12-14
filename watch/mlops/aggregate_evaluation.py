"""
Loads results from an evaluation and aggregates them

Ignore:

    # Real data
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

    python -m watch.mlops.aggregate_evaluation \
        --pipeline=bas \
        --root_dpath="$DVC_EXPT_DPATH/_testpipe"

"""

import scriptconfig as scfg


class AggregateEvluationConfig(scfg.DataConfig):
    root_dpath = scfg.Value('auto', help='Where do dump all results. If "auto", uses <expt_dvc_dpath>/dag_runs')
    pipeline = scfg.Value('joint_bas_sc', help='the name of the pipeline to run')


def main(cmdline=True, **kwargs):
    """
    Ignore:
        import watch
        data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        cmdline = 0
        kwargs = {
            'root_dpath': expt_dvc_dpath / '_testpipe',
            'pipeline': 'bas',
        }
    """
    config = AggregateEvluationConfig.legacy(cmdline=cmdline, data=kwargs)

    from watch.mlops import smart_pipeline
    from watch.utils import util_pattern
    dag = smart_pipeline.make_smart_pipeline(config['pipeline'])
    dag.print_graphs()
    dag.configure(config=None, root_dpath=config['root_dpath'])

    # Hard coded nodes of interest to gather. Should abstract later.
    eval_node_infos = [
        {'name': 'bas_pxl_eval', 'out_key': 'eval_pxl_fpath'},
        {'name': 'bas_poly_eval', 'out_key': 'eval_fpath'},
    ]
    eval_nodes = [x['name'] for x in eval_node_infos]
    eval_to_outkey = {x['name']: x['out_key'] for x in eval_node_infos}

    patterns = {
        'bas_pxl_id': '*',
        'bas_poly_id': '*',
        'bas_pxl_eval_id': '*',
        'bas_poly_eval_id': '*',
        'bas_poly_viz_id': '*',
    }

    import ubelt as ub
    for node in (ub.udict(dag.nodes) & eval_nodes).values():
        print(f'node.template_out_paths={node.template_out_paths}')
        out_key = eval_to_outkey[node.name]
        out_template = node.template_out_paths[out_key]
        pat = dag.root_dpath / out_template.format(**patterns)
        mpat = util_pattern.Pattern.coerce(pat)
        fpaths = list(mpat.paths())
        print('fpaths = {}'.format(ub.repr2(fpaths, nl=1)))

    for node_name, node in dag.nodes.items():
        ...

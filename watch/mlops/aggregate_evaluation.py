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
        from watch.mlops.aggregate_evaluation import *  # NOQA
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
    # eval_node_infos = [
    #     {'name': 'bas_pxl_eval', 'out_key': 'eval_pxl_fpath'},
    #     {'name': 'bas_poly_eval', 'out_key': 'eval_fpath'},
    # ]
    # eval_nodes = [x['name'] for x in eval_node_infos]
    # eval_to_outkey = {x['name']: x['out_key'] for x in eval_node_infos}

    # patterns = {
    #     'bas_pxl_id': '*',
    #     'bas_poly_id': '*',
    #     'bas_pxl_eval_id': '*',
    #     'bas_poly_eval_id': '*',
    #     'bas_poly_viz_id': '*',
    # }

    # for node_name, node in dag.nodes.items():
    #     node_matching_outputs(node)

    import parse
    out_template = dag.nodes['bas_pxl_eval'].outputs['eval_pxl_fpath'].template_value
    parser = parse.Parser(str(out_template))
    patterns = {n: '*' for n in parser.named_fields}
    pat = out_template.format(**patterns)
    mpat = util_pattern.Pattern.coerce(pat)
    fpaths = list(mpat.paths())

    import parse
    out_template = dag.nodes['bas_poly_eval'].outputs['eval_fpath'].template_value
    parser = parse.Parser(str(out_template))
    patterns = {n: '*' for n in parser.named_fields}
    pat = out_template.format(**patterns)
    mpat = util_pattern.Pattern.coerce(pat)
    fpaths = list(mpat.paths())

    from watch.mlops import smart_result_parser
    for fpath in fpaths:
        result = smart_result_parser.load_eval_trk_poly(fpath, None)
        print(result['metrics']['bas_faa_f1'])


# def node_matching_outputs(node):
#     from watch.utils import util_pattern
#     import ubelt as ub
#     # print(f'node.template_out_paths={node.template_out_paths}')
#     # out_key = eval_to_outkey[node.name]
#     found = {}
#     for out_key, out_template in node.template_out_paths.items():
#         out_template = node.template_out_paths[out_key]

#         # self._parse_pattern_attrs(self.templates[key], path)
#         pat = node.root_dpath / out_template.format(**patterns)
#         mpat = util_pattern.Pattern.coerce(pat)
#         fpaths = list(mpat.paths())
#         found[out_key] = fpaths

#     print(ub.map_vals(len, found))
#     # print('fpaths = {}'.format(ub.urepr(fpaths, nl=1)))

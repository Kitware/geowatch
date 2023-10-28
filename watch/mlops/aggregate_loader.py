"""
Logic for loading raw results from the MLops DAG root dir.

Used by ./aggregate.py
"""
import ubelt as ub
from kwutil import util_pattern
from kwutil import util_parallel
from watch.utils import util_dotdict
import parse
import json
from watch.mlops import smart_pipeline
from watch.mlops import smart_result_parser


def build_tables(root_dpath, pipeline, io_workers, eval_nodes,
                 cache_resolved_results):
    import pandas as pd
    from kwutil import util_progress
    dag = smart_pipeline.make_smart_pipeline(pipeline)
    dag.print_graphs()
    dag.configure(config=None, root_dpath=root_dpath)

    io_workers = util_parallel.coerce_num_workers(io_workers)
    print(f'io_workers={io_workers}')

    # patterns = {
    #     'bas_pxl_id': '*',
    #     'bas_poly_id': '*',
    #     'bas_pxl_eval_id': '*',
    #     'bas_poly_eval_id': '*',
    #     'bas_poly_viz_id': '*',
    # }

    # Hard coded nodes of interest to gather. Should abstract later.
    node_eval_infos = [
        {'name': 'bas_pxl_eval', 'out_key': 'eval_pxl_fpath',
         'result_loader': smart_result_parser.load_pxl_eval},
        {'name': 'sc_poly_eval', 'out_key': 'eval_fpath',
         'result_loader': smart_result_parser.load_sc_poly_eval},
        {'name': 'bas_poly_eval', 'out_key': 'eval_fpath',
         'result_loader': smart_result_parser.load_bas_poly_eval},

        {'name': 'sv_poly_eval', 'out_key': 'eval_fpath',
         'result_loader': smart_result_parser.load_iarpa_poly_eval},
    ]

    if eval_nodes is None:
        node_eval_infos_chosen = node_eval_infos
    else:
        lut = ub.udict({info['name']: info for info in node_eval_infos})
        node_eval_infos_chosen = list(lut.take(eval_nodes))

    from concurrent.futures import as_completed
    pman = util_progress.ProgressManager(backend='rich')
    # pman = util_progress.ProgressManager(backend='progiter')
    with pman:
        eval_type_to_results = {}

        eval_node_prog = pman.progiter(node_eval_infos_chosen, desc='Loading node results')

        for node_eval_info in eval_node_prog:
            node_name = node_eval_info['name']
            out_key = node_eval_info['out_key']
            # result_loader_fn = node_eval_info['result_loader']

            if node_name not in dag.nodes:
                continue

            node = dag.nodes[node_name]
            out_node = node.outputs[out_key]
            out_node_key = out_node.key

            fpaths = out_node_matching_fpaths(out_node)

            # Pattern match
            # node.template_out_paths[out_node.name]
            cols = {
                'index': [],
                'metrics': [],
                'requested_params': [],
                'resolved_params': [],
                'specified_params': [],
                'other': [],
                'fpath': [],
                # 'json_info': [],
            }

            executor = ub.Executor(mode='process', max_workers=io_workers)
            jobs = []
            submit_prog = pman.progiter(
                fpaths, desc=f'  * submit load jobs: {node_name}',
                transient=True)
            for fpath in submit_prog:
                job = executor.submit(load_result_worker, fpath, node_name,
                                      out_node_key,
                                      use_cache=cache_resolved_results)
                jobs.append(job)

            num_ignored = 0
            job_iter = as_completed(jobs)
            del jobs
            collect_prog = pman.progiter(
                job_iter, total=len(fpaths),
                desc=f'  * loading node results: {node_name}')
            for job in collect_prog:
                result = job.result()
                if result['requested_params']:
                    assert set(result.keys()) == set(cols.keys())
                    for k, v in result.items():
                        cols[k].append(v)
                else:
                    num_ignored += 1

            results = {
                'fpath': pd.DataFrame(cols['fpath'], columns=['fpath']),
                'index': pd.DataFrame(cols['index']),
                'metrics': pd.DataFrame(cols['metrics']),
                'requested_params': pd.DataFrame(cols['requested_params'], dtype=object),  # prevents nones from being read as nan
                'specified_params': pd.DataFrame(cols['specified_params']),
                'resolved_params': pd.DataFrame(cols['resolved_params'], dtype=object),
                'other': pd.DataFrame(cols['other']),
            }
            # print(results['resolved_params']['resolved_params.sc_poly.smoothing'])
            eval_type_to_results[node_name] = results

    return eval_type_to_results


def _lookup_result_loader(node_name):
    if node_name == 'bas_pxl_eval':
        result_loader_fn = smart_result_parser.load_pxl_eval
    elif node_name == 'bas_poly_eval':
        result_loader_fn = smart_result_parser.load_bas_poly_eval
    elif node_name == 'sv_poly_eval':
        result_loader_fn = smart_result_parser.load_bas_poly_eval
    elif node_name == 'sc_poly_eval':
        result_loader_fn = smart_result_parser.load_sc_poly_eval
    else:
        raise KeyError(node_name)
    return result_loader_fn


def load_result_worker(fpath, node_name, out_node_key, use_cache=True):
    """
    Ignore:
        fpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_1ad531cc/poly_eval.json')
        node_name = 'bas_poly_eval'
        out_node_key = 'bas_poly_eval.eval_fpath'
    """
    import json
    from watch.utils import util_json
    import safer
    fpath = ub.Path(fpath)

    resolved_json_fpath = fpath.parent / 'resolved_result_row_v7.json'

    if use_cache and resolved_json_fpath.exists():
        # Load the cached row data
        result = json.loads(resolved_json_fpath.read_text())
    else:
        node_dpath = fpath.parent

        node_dpath = ub.Path(node_dpath)
        # Read the requested config
        job_config_fpath = node_dpath / 'job_config.json'
        if job_config_fpath.exists():
            _requested_params = json.loads(job_config_fpath.read_text())
        else:
            _requested_params = {}

        requested_params = util_dotdict.DotDict(_requested_params).add_prefix('params')
        specified_params = {'specified.' + k: 1 for k in requested_params}

        # Read the resolved config
        # (Uses the DAG to trace the result lineage)
        try:
            flat = load_result_resolved(node_dpath)

            HACK_FOR_REGION_ID = True
            if HACK_FOR_REGION_ID:
                # Munge data to get the region ids we expect
                candidate_keys = list(flat.query_keys('region_ids'))
                region_ids = None
                for k in candidate_keys:
                    region_ids = flat[k]
                assert region_ids is not None
                import re
                region_pat = re.compile(r'[A-Z][A-Za-z]*_[A-Z]\d\d\d')
                region_ids = ','.join(list(region_pat.findall(region_ids)))

            resolved_params_keys = list(flat.query_keys('resolved_params'))
            metrics_keys = list(flat.query_keys('metrics'))
            resolved_params = flat & resolved_params_keys
            metrics = flat & metrics_keys

            other = flat - (resolved_params_keys + metrics_keys)

            index = {
                # 'type': out_node_key,
                'node': node_name,
                'region_id': region_ids,
            }
            result = {
                'fpath': fpath,
                'index': index,
                'metrics': metrics,
                'requested_params': requested_params,
                'resolved_params': resolved_params,
                'specified_params': specified_params,
                'other': other,
            }

            # Cache this resolved row data
            result = util_json.ensure_json_serializable(result)
        except Exception as ex:
            print(f'Failed to load results for: {node_name}')
            print(f'node_dpath={str(node_dpath)!r}')
            print('ex = {}'.format(ub.urepr(ex, nl=1)))
            raise

        with safer.open(resolved_json_fpath, 'w') as file:
            json.dump(result, file, indent=4)

    HACK_FOR_BAD_SV_REQUEST = 1
    if HACK_FOR_BAD_SV_REQUEST:
        if node_name == 'sv_poly_eval':
            import pandas as pd
            if pd.isnull(result['requested_params'].get('params.sv_crop.crop_src_fpath', None)):
                result['requested_params'] = None

    return result


def new_process_context_parser(proc_item):
    """
    Load parameters out of data saved by a ProcessContext object
    """
    tracker_name_pat = util_pattern.MultiPattern.coerce({
        'watch.cli.kwcoco_to_geojson',
        'watch.cli.run_tracker',
    })
    heatmap_name_pat = util_pattern.MultiPattern.coerce({
        'watch.tasks.fusion.predict',
    })
    pxl_eval_pat = util_pattern.MultiPattern.coerce({
        'watch.tasks.fusion.evaluate',
    })
    proc_item = smart_result_parser._handle_process_item(proc_item)
    props = proc_item['properties']

    # Node-specific hacks
    params = props['config']
    if tracker_name_pat.match(props['name']):
        params.update(**json.loads(params.pop('track_kwargs', '{}')))
    elif heatmap_name_pat.match(props['name']):
        params.pop('datamodule_defaults', None)
    elif pxl_eval_pat.match(props['name']):
        from watch.tasks.fusion import evaluate
        # We can resolve the params to a dictionary in this instance
        if isinstance(params, list) or 'true_dataset' not in params:
            args = props['args']
            params = evaluate.SegmentationEvalConfig().load(cmdline=args).to_dict()

    resources = smart_result_parser.parse_resource_item(proc_item, add_prefix=False)

    output = {
        # TODO: better name for this
        'context': {
            'task': props['name'],
            'uuid': props.get('uuid', None),
            'start_timestamp': props.get('start_timestamp', None),
            'stop_timestamp': props.get('stop_timestamp', props.get('end_timestamp', None)),
        },
        'resolved_params': params,
        'resources': resources,
        'machine': props.get('machine', {}),
    }
    return output


def load_result_resolved(node_dpath):
    """
    Recurse through the DAG filesytem structure and load resolved
    configurations from each step.

    Args:
        node_dpath (str | PathLike):
            the path to the evaluation node directory. The specific type of
            evaluation node must have a known (currently hard-coded) condition
            in this function that knows how to parse it.

    Returns:
        Dict - flat_resolved - a flat dot-dictionary with resolved params

    TODO:
        Some mechanism to let the user specify how to parse an evaluation node
        of a given type.

    Ignore:
        from watch.mlops.aggregate import *  # NOQA
        node_dpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_1ad531cc')
        got = load_result_resolved(node_dpath)
        node_dpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_pxl_eval/bas_pxl_eval_id_6028edfe/')
        node_dpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_timekernel_test_drop4/eval/flat/bas_pxl_eval/bas_pxl_eval_id_5d38c6b3')
    """
    # from watch.utils.util_dotdict import explore_nested_dict
    node_dpath = ub.Path(node_dpath)
    node_type_dpath = node_dpath.parent
    node_type = node_type_dpath.name

    if node_type in {'sc_pxl', 'bas_pxl'}:
        pat = util_pattern.Pattern.coerce(node_dpath / 'pred.kwcoco.*')
        fpath = list(pat.paths())[0]
        bas_pxl_info = smart_result_parser.parse_json_header(fpath)
        proc_item = smart_result_parser.find_pred_pxl_item(bas_pxl_info)
        nest_resolved = new_process_context_parser(proc_item)
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(node_type, index=1)

        # Hack: we should recurse into the model itself to introspect this, but
        # this is fine for now.
        fit_node_type = node_type + '_fit'
        fit_config = proc_item['properties']['extra']['fit_config']
        fit_config = smart_result_parser.relevant_fit_config(fit_config, add_prefix=False)
        fit_nested = {
            'context': {'task': 'watch.tasks.fusion.fit'},
            'resolved_params': fit_config,
            'resources': {},
            'machine': {},
        }
        flat_fit_resolved = util_dotdict.DotDict.from_nested(fit_nested)
        flat_fit_resolved = flat_fit_resolved.insert_prefix(fit_node_type, index=1)
        flat_resolved |= flat_fit_resolved

    elif node_type in {'bas_poly', 'sc_poly'}:
        pat = util_pattern.Pattern.coerce(node_dpath / 'poly.kwcoco.*')
        fpath = list(pat.paths())[0]
        try:
            bas_poly_info = smart_result_parser.parse_json_header(fpath)
        except Exception:
            # There are some cases where the kwcoco file was clobbered,
            # but we can work around by using a manifest file.
            pat = util_pattern.Pattern.coerce(node_dpath / '*_manifest.json')
            fpath = list(pat.paths())[0]
            bas_poly_info = smart_result_parser.parse_json_header(fpath)

        proc_item = smart_result_parser.find_track_item(bas_poly_info)
        nest_resolved = new_process_context_parser(proc_item)
        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(node_type, index=1)

    elif node_type in {'bas_poly_eval', 'sc_poly_eval', 'sv_poly_eval'}:
        fpath = node_dpath / 'poly_eval.json'
        iarpa_result = smart_result_parser.load_iarpa_evaluation(fpath)
        proc_item = smart_result_parser.find_metrics_framework_item(
            iarpa_result['iarpa_json']['info'])
        nest_resolved = new_process_context_parser(proc_item)
        nest_resolved['metrics'] = iarpa_result['metrics']

        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved['context.region_ids'] = iarpa_result['iarpa_json']['region_ids']
        flat_resolved = flat_resolved.insert_prefix(node_type, 1)

    elif node_type in {'bas_pxl_eval', 'sc_pxl_eval'}:
        fpath = node_dpath / 'pxl_eval.json'
        info = smart_result_parser.load_pxl_eval(fpath, with_param_types=False)
        metrics = info['metrics']

        proc_item = smart_result_parser.find_pxl_eval_item(
            info['json_info']['meta']['info'])

        nest_resolved = new_process_context_parser(proc_item)
        # Hack for region ids
        nest_resolved['context']['region_ids'] = ub.Path(nest_resolved['resolved_params']['true_dataset']).name.split('.')[0]
        nest_resolved['metrics'] = metrics

        flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        flat_resolved = flat_resolved.insert_prefix(node_type, 1)

    elif node_type in {'valicrop', 'sv_crop'}:

        # TODO: parse resolved params
        # nest_resolved = {}
        # flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
        # flat_resolved = flat_resolved.insert_prefix(node_type, index=1)
        node_process_name = 'coco_align'
        fpath = node_dpath / 'sv_crop.kwcoco.zip'
        if not fpath.exists():
            fpath = node_dpath / 'valicrop.kwcoco.zip'
        flat_resolved = _generalized_process_flat_resolved(fpath, node_process_name, node_type)

    elif node_type in {'sc_crop'}:
        # TODO: parse resolved params
        node_process_name = 'coco_align'
        fpath = node_dpath / 'sitecrop.kwcoco.zip'
        flat_resolved = _generalized_process_flat_resolved(fpath, node_process_name, node_type)

    elif node_type in {'cluster_sites'}:
        # TODO: write out resolved params in cluster sites
        flat_resolved = {}

    elif node_type in {'sv_dino_boxes'}:
        node_process_name = 'box.predict'
        fpath = node_dpath / 'pred_boxes.kwcoco.zip'
        flat_resolved = _generalized_process_flat_resolved(fpath, node_process_name, node_type)

    elif node_type in {'sv_dino_filter'}:
        node_process_name = 'watch.tasks.dino_detector.building_validator'
        fpath = node_dpath / 'out_site_manifest.json'
        flat_resolved = _generalized_process_flat_resolved(fpath, node_process_name, node_type)
    # TODO: it would be nice if just declaring a node gave us this information.
    elif node_type in {'sv_depth_score'}:
        from watch.mlops import smart_pipeline
        fpath = node_dpath / smart_pipeline.SV_DepthPredict.out_paths['out_kwcoco']
        node_process_name = 'watch.tasks.depth_pcd.score_tracks'
        flat_resolved = _generalized_process_flat_resolved(fpath, node_process_name, node_type)
    elif node_type in {'sv_depth_filter'}:
        from watch.mlops import smart_pipeline
        fpath = node_dpath / smart_pipeline.SV_DepthFilter.out_paths['output_site_manifest_fpath']
        node_process_name = 'watch.tasks.depth_pcd.filter_tracks'
        flat_resolved = _generalized_process_flat_resolved(fpath, node_process_name, node_type)
    else:
        raise NotImplementedError(node_type)

    predecessor_dpath = node_dpath / '.pred'
    for predecessor_node_type_dpath in predecessor_dpath.glob('*'):
        # predecessor_node_type = predecessor_node_type_dpath.name
        for predecessor_node_dpath in predecessor_node_type_dpath.glob('*'):
            if predecessor_node_dpath.exists():
                flat_resolved |= load_result_resolved(predecessor_node_dpath)

    return flat_resolved


def _generalized_process_flat_resolved(fpath, node_process_name, node_type):
    """
    Parses info out of json files where we conform to a pattern that the top
    level json object has an "info" section as list of ProcessContext info
    objects. We grab the context of a metric-less node there. We do need to
    know the filepath of this json file and what the name used in process
    context was.
    """
    info = smart_result_parser.parse_json_header(fpath)
    proc_item = smart_result_parser.find_info_items(info, {'process'}, {node_process_name})
    items = list(proc_item)
    assert len(items) == 1
    proc_item = items[0]
    nest_resolved = new_process_context_parser(proc_item)
    flat_resolved = util_dotdict.DotDict.from_nested(nest_resolved)
    flat_resolved = flat_resolved.insert_prefix(node_type, index=1)
    return flat_resolved


def out_node_matching_fpaths(out_node):
    out_template = out_node.template_value
    parser = parse.Parser(str(out_template))
    patterns = {n: '*' for n in parser.named_fields}
    pat = out_template.format(**patterns)
    mpat = util_pattern.Pattern.coerce(pat)
    fpaths = list(mpat.paths())
    return fpaths


if 1:
    import numpy as np
    if np.bool_ is not bool:
        # Hack for a ubelt issue
        @ub.hash_data.register(np.bool_)
        def _hashnp_bool(data):
            from ubelt.util_hash import _int_to_bytes
            # warnings.warn('Hashing ints is slow, numpy is preferred')
            hashable = _int_to_bytes(bool(data))
            # hashable = data.to_bytes(8, byteorder='big')
            prefix = b'INT'
            return prefix, hashable

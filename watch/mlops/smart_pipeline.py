"""
This is a very task-specific file containing logic to parse fusion pipeline
metrics for BAS and SC.


Generate:
    import liberator
    lib = liberator.Liberator()
    import networkx as nx
    lib.add_dynamic(nx.DiGraph)
    print(ub.highlight_code(lib.current_sourcecode()))
"""

import numpy as np
import warnings
import json
import ubelt as ub
import io
import pandas as pd


def get_column_meanings():
    return [
        {'name': 'raw', 'help': 'A full path to a file on disk that contains this info'},
        {'name': 'dvc', 'help': 'A path to a DVC sidecar file if it exists.'},
        {'name': 'type', 'help': 'The type of the row'},
        {'name': 'step', 'help': 'The number of steps taken by the most recent training run associated with the row'},
        {'name': 'total_steps', 'help': 'An estimate of the total number of steps the model associated with the row took over all training runs.'},
        {'name': 'model', 'help': 'The name of the learned model associated with this row'},
        # {'name': 'test_dset', 'help': 'The name of the test dataset used to compute a metric associated with this row'},
        {'name': 'test_trk_dset', 'help': 'The name of the test BAS dataset used to compute a metric associated with this row'},
        {'name': 'test_act_dset', 'help': 'The name of the test SC dataset used to compute a metric associated with this row'},

        {'name': 'expt', 'help': 'The name of the experiment, i.e. training session that might have made several models'},
        {'name': 'dataset_code', 'help': 'The higher level dataset code associated with this row'},

        {'name': 'pred_cfg', 'help': 'A hash of the configuration used for pixel heatmap prediction'},
        {'name': 'trk_cfg', 'help': 'A hash of the configuration used for BAS tracking'},
        {'name': 'act_cfg', 'help': 'A hash of the configuration used for SC classification'},

        {'name': 'total_steps', 'help': 'An estimate of the total number of steps the model associated with the row took over all training runs.'},
    ]


def partition_params():
    pass


def parse_json_header(fpath):
    """
    Ideally the information we need is in the first few bytes of the json file
    """
    pass


def trace_json_lineage(fpath):
    """
    We will expect a json file to contain a top-level "info" section that
    indicates how it is derived.

    fpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/trk/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data.kwcoco/trk_pxl_b788335d/trk_poly_f2218f0b/tracks.kwcoco.json'
    """
    from watch.utils import ijson_ext
    with open(fpath, 'r') as file:
        # We only expect there to be one info section
        # file.seek(0)
        # iii = ijson_ext.items(file, '')
        info_section_iter = ijson_ext.items(file, prefix='info')
        info_section = next(info_section_iter)

    # TODO:
    # uniqify by uuid
    for proc in list(find_info_items(info_section, 'process')):
        name = proc['properties']['name']
        if name not in {'coco_align_geotiffs'}:
            print(f'name={name}')
            print(proc['properties']['start_timestamp'])
            print(proc['properties']['emissions']['run_id'])
            print('proc = {}'.format(ub.repr2(proc, nl=2)))
        # print(proc['properties']['name'])


# def trace_kwcoco_lineage(fpath):
def load_iarpa_evaluation(fpath):
    print(f'fpath={fpath}')
    iarpa_info = _load_json(fpath)
    metrics = {}
    if 'best_bas_rows' in iarpa_info:
        best_bas_rows = pd.read_json(io.StringIO(json.dumps(iarpa_info['best_bas_rows'])), orient='table')
        bas_row = best_bas_rows.loc['__macro__'].reset_index().iloc[0]

        metrics.update({
            'bas_tp': bas_row['tp sites'],
            'bas_fp': bas_row['fp sites'],
            'bas_fn': bas_row['fn sites'],
            'bas_ntrue': bas_row['total sites'],
            'bas_npred': bas_row['proposed slices'],
            'bas_ppv': bas_row['precision'],
            'bas_tpr': bas_row['recall (PD)'],
        })
        metrics.update({
            'bas_f1': bas_row['F1'],
            'rho': bas_row['rho'],
            'tau': bas_row['tau'],
        })
    if 'sc_df' in iarpa_info:
        sc_df = pd.read_json(io.StringIO(json.dumps(iarpa_info['sc_df'])), orient='table')
        metrics.update({
            # 'mean_f1': sc_df.loc['F1'].mean(),
            'macro_f1': sc_df.loc['__macro__']['F1'].mean(),
            'micro_f1': sc_df.loc['__micro__']['F1'].mean(),
            'macro_f1_siteprep': sc_df.loc['__macro__', 'Site Preparation']['F1'],
            'macro_f1_active': sc_df.loc['__macro__', 'Site Preparation']['F1'],
        })
    return metrics, iarpa_info


def load_eval_trk_poly(fpath, expt_dvc_dpath):
    arg_prefix = 'trk.'
    metrics, iarpa_info = load_iarpa_evaluation(fpath)
    tracker_info = iarpa_info['parent_info']
    param_types = parse_tracker_params(tracker_info, expt_dvc_dpath, arg_prefix=arg_prefix)
    extra_attrs = _add_prefix(arg_prefix + 'poly.metrics.', metrics)
    info = {
        'fpath': fpath,
        'metrics': metrics,
        'param_types': param_types,
        'other': {
            'extra_attrs': extra_attrs,
        },
        'json_info': iarpa_info,
    }
    return info


def load_eval_act_poly(fpath, expt_dvc_dpath):
    arg_prefix = 'act.'
    metrics, iarpa_info = load_iarpa_evaluation(fpath)
    tracker_info = iarpa_info['parent_info']
    param_types = parse_tracker_params(tracker_info, expt_dvc_dpath, arg_prefix=arg_prefix)
    # Hack to grab information that we should have already had.
    HACK_HANDLE_CROPPED_AND_TRACK_PARAMS = 1
    if HACK_HANDLE_CROPPED_AND_TRACK_PARAMS:
        trk_param_types, extra_attrs = _handle_crop_and_trk_params(param_types, expt_dvc_dpath)
        param_types.update(trk_param_types)
    else:
        extra_attrs = {}
    extra_attrs.update(_add_prefix('act.poly.metrics.', metrics))
    info = {
        'fpath': fpath,
        'metrics': metrics,
        'param_types': param_types,
        'other': {
            'extra_attrs': extra_attrs,
        },
        'json_info': iarpa_info,
    }
    return info


def _handle_crop_and_trk_params(param_types, expt_dvc_dpath):
    from watch.mlops import expt_manager
    crop_fpath = param_types['act.pxl']['act.pxl.test_dataset']
    state = expt_manager.ExperimentState('*', '*')
    crop_attrs = ub.udict(state._parse_pattern_attrs(
        state.templates['crop_fpath'], crop_fpath))

    crop_dataset = _load_json(crop_fpath)
    crop_item = list(
        find_info_items(crop_dataset['info'], 'process', 'coco_align_geotiffs'))[-1]

    # This is the path to either truth or the tracks we cropped from
    region_fpath = crop_item['properties']['args']['regions']

    try:
        trk_attrs = ub.udict(state._parse_pattern_attrs(
            state.templates['pred_trk_poly_site_summaries_fpath'],
            region_fpath))
    except Exception:
        trk_attrs = ub.udict(state._parse_pattern_attrs(
            state.templates['pred_trk_poly_sites_fpath'],
            region_fpath))

    extra_attrs = ub.udict(crop_attrs) | ub.udict(trk_attrs)

    trk_poly_data = _load_json(region_fpath)
    trk_poly_info = trk_poly_data['info']
    trk_param_types = parse_tracker_params(trk_poly_info, expt_dvc_dpath, arg_prefix='trk.')
    return trk_param_types, extra_attrs


def parse_tracker_params(tracker_info, expt_dvc_dpath, arg_prefix=''):
    """
    Note:
        This is tricky because we need to find a way to differentiate if this
        was a trk or bas tracker.
    """
    track_item = find_track_item(tracker_info)
    track_item = _handle_process_item(track_item)

    if 'extra' in track_item['properties']:
        pred_info = track_item['properties']['extra']['pred_info']
    else:
        raise AssertionError

    param_types = parse_pred_pxl_params(
        pred_info, expt_dvc_dpath, arg_prefix=arg_prefix)

    poly_resources = parse_resource_item(track_item, arg_prefix=(arg_prefix + 'poly.'))
    param_types[arg_prefix + 'poly.resource'] = poly_resources

    track_args = track_item['properties']['config']
    track_config = relevant_track_config(track_args, arg_prefix=arg_prefix)
    param_types[arg_prefix + 'poly'] = track_config
    return param_types


def _handle_process_item(item):
    """
    Json data written by the process context has changed over time slightly.
    Consolidate different usages until a consistent API and usage patterns are
    established.

    """
    assert item['type'] in {'process', 'process_context'}
    props = item['properties']

    needs_modify = 0

    config = props.get('config', None)
    args = props.get('args', None)
    if config is None:
        # Use args if config is not available
        config = args
        needs_modify = True

    FIX_BROKEN_SCRIPTCONFIG_HANDLING = 1
    if FIX_BROKEN_SCRIPTCONFIG_HANDLING:
        if '_data' in config:
            config = config['_data']
            needs_modify = True
        if '_data' in args:
            args = args['_data']
            needs_modify = True

    assert 'pred_info' not in item, 'should be in extra instead'

    if needs_modify:
        import copy
        item = copy.deepcopy(item)
        item['properties']['config'] = config
        item['properties']['args'] = args

    return item


def load_pxl_eval(fpath, expt_dvc_dpath=None, arg_prefix=''):
    from kwcoco.coco_evaluator import CocoSingleResult
    from watch.utils import util_pattern
    # from watch.utils import result_analysis
    # from watch.utils import util_time
    measure_info = _load_json(fpath)

    meta = measure_info['meta']

    pred_info = meta['info']
    dvc_dpath = expt_dvc_dpath
    param_types = parse_pred_pxl_params(pred_info, dvc_dpath, arg_prefix=arg_prefix)

    predict_args = param_types[arg_prefix + 'pxl']
    if predict_args is None:
        raise Exception('no prediction metadata')

    salient_measures = measure_info['nocls_measures']
    class_measures = measure_info['ovr_measures']

    # HACK: fixme
    coi_pattern = util_pattern.MultiPattern.coerce(
        ['Active Construction', 'Site Preparation'], hint='glob')

    class_metrics = []
    coi_metrics = []
    for catname, bin_measure in class_measures.items():
        class_row = {}
        class_row['AP'] = bin_measure['ap']
        class_row['AUC'] = bin_measure['auc']
        class_row['APUC'] = np.nanmean([bin_measure['ap'], bin_measure['auc']])
        class_row['catname'] = catname
        if coi_pattern.match(catname):
            coi_metrics.append(class_row)
        class_metrics.append(class_row)

    class_aps = [r['AP'] for r in class_metrics]
    class_aucs = [r['AUC'] for r in class_metrics]
    coi_aps = [r['AP'] for r in coi_metrics]
    coi_aucs = [r['AUC'] for r in coi_metrics]
    coi_catnames = [r['catname'] for r in coi_metrics]

    metrics = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        metrics['class_mAP'] = np.nanmean(class_aps) if len(class_aps) else np.nan
        metrics['class_mAUC'] = np.nanmean(class_aucs) if len(class_aucs) else np.nan
        metrics['class_mAPUC'] = np.nanmean([metrics['class_mAUC'], metrics['class_mAP']])

        metrics['coi_mAP'] = np.nanmean(coi_aps) if len(coi_aps) else np.nan
        metrics['coi_mAUC'] = np.nanmean(coi_aucs) if len(coi_aucs) else np.nan
        metrics['coi_mAPUC'] = np.nanmean([metrics['coi_mAUC'], metrics['coi_mAP']])

        metrics['salient_AP'] = salient_measures['ap']
        metrics['salient_AUC'] = salient_measures['auc']
        metrics['salient_APUC'] = np.nanmean([metrics['salient_AP'], metrics['salient_AUC']])

    for class_row in class_metrics:
        metrics[class_row['catname'] + '_AP'] = class_row['AP']
        metrics[class_row['catname'] + '_AUC'] = class_row['AUC']

    result = CocoSingleResult.from_json(measure_info)

    HACK_HANDLE_CROPPED_AND_TRACK_PARAMS = 1
    if HACK_HANDLE_CROPPED_AND_TRACK_PARAMS and arg_prefix == 'act.':
        trk_param_types, extra_attrs = _handle_crop_and_trk_params(
            param_types, expt_dvc_dpath)
        param_types.update(trk_param_types)

    else:
        extra_attrs = {}
    extra_attrs.update(_add_prefix(arg_prefix + 'pxl.metrics.', metrics))

    info = {
        'fpath': fpath,
        'metrics': metrics,
        'param_types': param_types,
        'other': {
            'result': result,
            'extra_attrs': extra_attrs,
            'coi_catnames': ','.join(sorted(coi_catnames)),
            # 'sc_cm': sc_cm,
            # 'sc_df': sc_df,
        },
        'json_info': measure_info,
    }
    return info


class Found(Exception):
    pass


def resolve_cross_machine_path(path, dvc_dpath=None):
    """
    HACK

    Attempt to determine what the local path to a file/directry would be
    if it exists on this machine. This assumes the path is something
    that was checked into DVC.

    Args:
        dvc_dpath : the preferred dvc dpath to associate the file with
            in case the older one points to multiple.
    """
    # pkg_dvc = SimpleDVC.find_root(package_fpath).resolve()
    # SimpleDVC.find_root(package_fpath).resolve()
    path = ub.Path(path)
    needs_resolve = not path.exists()
    if not needs_resolve:
        if dvc_dpath is not None:
            needs_resolve = not path.is_relative_to(dvc_dpath)

    if needs_resolve:
        expected_dnames = [
            'smart_watch_dvc',
            'smart_watch_dvc-hdd',
            'smart_watch_dvc-ssd',
        ]

        found_idx = None
        for dname in expected_dnames:
            try:
                idx = path.parts.index(dname)
            except ValueError:
                pass
            else:
                found_idx = idx
                break

        if found_idx is not None:
            # import watch
            # dvc_dpath = watch.find_smart_dvc_dpath()
            pname = ub.Path(*path.parts[idx + 1:])
            pname_dvc = pname.augment(tail='.dvc')
            cwd = ub.Path('.').absolute()
            candidates = []
            if dvc_dpath is not None:
                candidates.extend([
                    dvc_dpath / pname,
                    dvc_dpath / pname_dvc,
                ])
            candidates.extend([
                cwd / pname,
                cwd / pname_dvc,
            ])
            found = None
            try:
                for cand_path in candidates:
                    if cand_path.exists():
                        found = cand_path
                        if found.name.endswith('.dvc'):
                            found = found.augment(ext='')
                        raise Found
            except Found:
                pass
            if found:
                return found
    return path


@ub.memoize
def global_ureg():
    import pint
    ureg = pint.UnitRegistry()
    return ureg


def _add_prefix(prefix, dict_):
    return {prefix + k: v for k, v in dict_.items()}


def relevant_pred_pxl_config(pred_args, dvc_dpath, arg_prefix=''):
    # TODO: better way of inferring what params are relevant
    # This should be metadata a scriptconfig object can hold.
    pred_config = {}
    pred_config['tta_fliprot'] = pred_args.get('tta_fliprot', 0)
    pred_config['tta_time'] = pred_args.get('tta_time', 0)
    pred_config['chip_overlap'] = pred_args['chip_overlap']
    pred_config['input_space_scale'] = pred_args.get('input_space_scale', None)
    pred_config['window_space_scale'] = pred_args.get('window_space_scale', None)
    pred_config['output_space_scale'] = pred_args.get('output_space_scale', None)
    pred_config['time_span'] = pred_args.get('time_span', None)
    pred_config['time_sampling'] = pred_args.get('time_sampling', None)
    pred_config['time_steps'] = pred_args.get('time_steps', None)
    pred_config['chip_dims'] = pred_args.get('chip_dims', None)
    pred_config['set_cover_algo'] = pred_args.get('set_cover_algo', None)
    pred_config['resample_invalid_frames'] = pred_args.get('resample_invalid_frames', None)
    pred_config['use_cloudmask'] = pred_args.get('use_cloudmask', None)
    package_fpath = pred_args['package_fpath']
    test_dataset = pred_args['test_dataset']
    if dvc_dpath is not None:
        package_fpath = resolve_cross_machine_path(package_fpath, dvc_dpath)
        test_dataset = resolve_cross_machine_path(test_dataset, dvc_dpath)

    # pred_config['model_fpath'] = package_fpath
    # pred_config['in_dataset'] = test_dataset
    pred_config['package_fpath'] = package_fpath
    pred_config['test_dataset'] = test_dataset

    pred_config['properties.model_name'] = model_name = ub.Path(package_fpath).name
    pred_config['properties.dataset_name'] = str(ub.Path(*test_dataset.parts[-2:]))

    # Hack to get the epoch/step/expt_name
    try:
        epoch = int(model_name.split('epoch=')[1].split('-')[0])
    except Exception:
        epoch = -1
    try:
        step = int(model_name.split('step=')[1].split('-')[0])
    except Exception:
        step = -1
    try:
        expt_name = model_name.split('_epoch=')[0]
    except Exception:
        expt_name = '?'
        # expt_name = predict_args[expt_name]

    pred_config['properties.step'] = step
    pred_config['properties.epoch'] = epoch
    pred_config['properties.expt_name'] = expt_name
    pred_config = _add_prefix(arg_prefix + 'pxl.', pred_config)
    return pred_config


def relevant_fit_config(fit_config, arg_prefix=''):
    ignore_params = {
        'default_root_dir', 'enable_progress_bar'
        'prepare_data_per_node', 'enable_model_summary', 'checkpoint_callback',
        'detect_anomaly', 'gpus', 'terminate_on_nan', 'train_dataset',
        'workdir', 'config', 'num_workers', 'amp_backend',
        'enable_progress_bar', 'flush_logs_every_n_steps',
        'enable_checkpointing', 'prepare_data_per_node', 'amp_level',
        'vali_dataset', 'test_dataset', 'package_fpath', 'num_draw',
        'num_nodes', 'num_processes', 'num_sanity_val_steps',
        'overfit_batches', 'process_position',
        'reload_dataloaders_every_epoch', 'reload_dataloaders_every_n_epochs',
        'replace_sampler_ddp', 'sync_batchnorm', 'torch_sharing_strategy',
        'torch_start_method', 'val_check_interval', 'weights_summary',
        'auto_lr_find', 'auto_select_gpus', 'auto_scale_batch_size', 'benchmark',
        'check_val_every_n_epoch', 'draw_interval', 'eval_after_fit', 'fast_dev_run',
        'limit_predict_batches', 'limit_test_batches', 'limit_train_batches',
        'limit_val_batches', 'log_every_n_steps', 'logger',
        'move_metrics_to_cpu', 'multiple_trainloader_mode',
    }
    from scriptconfig import smartcast
    # hack, rectify different values of known parameters that mean the
    # same thing
    fit_config2 = ub.dict_diff(fit_config, ignore_params)
    for k, v in fit_config2.items():
        if k not in {'channels', 'init'}:
            v2 = smartcast.smartcast(v)
            if isinstance(v2, list):
                # Dont coerce into a list
                v2 = v
            fit_config2[k] = v2
        else:
            fit_config2[k] = v

    if 'init' in fit_config2:
        # hack to make init only use the filename
        fit_config2['init'] = fit_config2['init'].split('/')[-1]
    fit_config2 = _add_prefix(arg_prefix + 'fit.', fit_config2)
    return fit_config2


def relevant_track_config(track_args, arg_prefix=''):
    track_config = json.loads(track_args['track_kwargs'])
    track_config = _add_prefix(arg_prefix + 'poly.', track_config)
    return track_config


def parse_resource_item(item, arg_prefix=''):
    from watch.utils import util_time

    resources = {}

    ureg = global_ureg()
    pred_prop = item['properties']

    start_time = util_time.coerce_datetime(pred_prop.get('start_timestamp', None))
    end_time = util_time.coerce_datetime(pred_prop.get('end_timestamp', pred_prop.get('stop_timestamp', None)))
    iters_per_second = pred_prop.get('iters_per_second', None)
    total_hours = (end_time - start_time).total_seconds() / (60 * 60)
    resources['total_hours'] = total_hours
    if iters_per_second is not None:
        resources['iters_per_second'] = iters_per_second

    try:
        vram = pred_prop['device_info']['allocated_vram']
        vram_gb = ureg.parse_expression(f'{vram} bytes').to('gigabytes').m
        resources['vram_gb'] = vram_gb
    except KeyError:
        ...

    hardware_parts = []

    import re
    cpu_name = pred_prop['machine']['cpu_brand']
    cpu_name = re.sub('.*Gen Intel.R. Core.TM. ', '', cpu_name)
    resources['cpu_name'] = cpu_name
    hardware_parts.append(cpu_name)

    try:
        gpu_name = pred_prop['device_info']['device_name']
        resources['gpu_name'] = gpu_name
        hardware_parts.append(gpu_name)
    except KeyError:
        ...

    if 'emissions' in pred_prop:
        co2_kg = pred_prop['emissions']['co2_kg']
        kwh = pred_prop['emissions']['total_kWH']
        resources['co2_kg'] = co2_kg
        resources['kwh'] = kwh

    if 'disk_info' in pred_prop:
        disk_type = pred_prop['disk_info']['filesystem']
        resources['disk_type'] = disk_type

    resources['hardware'] = ' '.join(hardware_parts)
    resources = _add_prefix(arg_prefix + 'resource.', resources)
    return resources


def find_pred_pxl_item(pred_info):
    pred_items = list(find_info_items(pred_info, 'process', 'watch.tasks.fusion.predict'))
    assert len(pred_items) == 1
    pred_item = pred_items[0]
    return pred_item


def find_info_items(info, query_type, query_name=None):
    for item in info:
        if item['type'] == query_type:
            if query_name is None or item['properties']['name'] == query_name:
                yield item


def parse_pred_pxl_params(pred_info, expt_dvc_dpath, arg_prefix=''):
    pred_item = find_pred_pxl_item(pred_info)
    pred_item = _handle_process_item(pred_item)

    # NOTE: the place where measure are stored has changed to be inside
    # the pred item.
    assert not list(find_info_items(pred_info, 'measure', None))

    meta = {'start_time': None}
    resources = {}
    # New code should have measures inside the pred item
    fit_config = pred_item['properties']['extra']['fit_config']
    meta['start_time'] = pred_item['properties']['start_timestamp']
    meta['end_time'] = pred_item['properties']['stop_timestamp']
    meta['start_timestamp'] = pred_item['properties']['start_timestamp']
    meta['stop_timestamp'] = pred_item['properties']['stop_timestamp']

    resources = parse_resource_item(pred_item, arg_prefix=(arg_prefix + 'pxl.'))

    pred_pxl_config = pred_item['properties']['config']
    pred_pxl_config = relevant_pred_pxl_config(
        pred_pxl_config, expt_dvc_dpath, arg_prefix=arg_prefix)
    fit_config = relevant_fit_config(fit_config, arg_prefix=arg_prefix)

    param_types = {
        arg_prefix + 'fit': fit_config,
        arg_prefix + 'pxl': pred_pxl_config,
        arg_prefix + 'pxl.resource': resources,
        arg_prefix + 'pxl.meta': _add_prefix(arg_prefix + 'pxl.meta.', meta),
    }
    return param_types


@ub.memoize
def _load_json(fpath):
    # memo hack for development
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data


def find_track_item(tracker_info):
    track_items = list(find_info_items(tracker_info, 'process', 'watch.cli.kwcoco_to_geojson'))
    assert len(track_items) == 1
    track_item = track_items[0]
    return track_item


def shrink_channels(x):
    import kwcoco
    aliases = {
        'blue': 'B',
        'red': 'R',
        'green': 'G',
        'nir': 'N',
        'swir16': 'S',
        'swir22': 'H',
    }
    for idx, part in enumerate('forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field'.split('|')):
        aliases[part] =  'land.{}'.format(idx)
    stream_parts = []
    sensorchan = kwcoco.SensorChanSpec.coerce(x)
    # spec = kwcoco.ChannelSpec.coerce(x)
    for stream in sensorchan.streams():
        fused_parts = []
        for c in stream.chans.as_list():
            c = aliases.get(c, c)
            c = c.replace('matseg_', 'matseg.')
            fused_parts.append(c)
        fused = '|'.join(fused_parts)
        fused = fused.replace('B|G|R|N', 'BGRN')
        fused = fused.replace('B|G|R|N|S|H', 'BGRNSH')
        fused = fused.replace('R|G|B', 'RGB')
        fused = fused.replace('B|G|R', 'BGR')
        stream_parts.append(stream.sensor.spec + ':' + fused)
    new = ','.join(stream_parts)
    x = kwcoco.SensorChanSpec.coerce(new).concise().spec
    return x


def is_teamfeat(sensorchan):
    """
    Check if the sensorchan spec contains a hard coded value we know is a team
    feature
    """
    import math
    unique_chans = sum([s.chans for s in sensorchan.streams()]).fuse().to_set()
    if isinstance(unique_chans, float) and math.isnan(unique_chans):
        return False
    return any([a in unique_chans for a in ['depth', 'invariant', 'invariants', 'matseg', 'land']])

"""
This is a very task-specific file containing logic to parse fusion pipeline
metrics for BAS and SC.

Used by ./aggregate_loader.py
"""

import numpy as np
import warnings
import json
import ubelt as ub
import io
# import xdev
import pandas as pd
import re
from kwutil import util_time
from kwutil import util_pattern


@ub.memoize
def parse_json_header(fpath):
    """
    Ideally the information we need is in the first few bytes of the json file
    """
    from geowatch.utils import ijson_ext
    import zipfile
    if zipfile.is_zipfile(fpath):
        # We have a compressed json file, but we can still read the header
        # fairly quickly.
        zfile = zipfile.ZipFile(fpath)
        names = zfile.namelist()
        assert len(names) == 1
        member = names[0]
        # Stream the header directly from the zipfile.
        file = zfile.open(member, 'r')
    else:
        # Normal json file
        file = open(fpath, 'r')

    with file:
        # import ijson
        # We only expect there to be one info section
        # try:
        #     # Try our extension if the main library fails (due to NaN)
        #     info_section_iter = ijson.items(file, prefix='info')
        #     info_section = next(info_section_iter)
        # except ijson.IncompleteJSONError:
        # Try our extension if the main library fails (due to NaN)
        # file.seek(0)

        # Nans are too frequent, only use our extension
        info_section_iter = ijson_ext.items(file, prefix='info')
        info_section = next(info_section_iter)
    return info_section


def trace_json_lineage(fpath):
    """
    We will expect a json file to contain a top-level "info" section that
    indicates how it is derived.

    fpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/trk/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data.kwcoco/trk_pxl_b788335d/trk_poly_f2218f0b/tracks.kwcoco.json'
    """

    info_section = parse_json_header(fpath)

    # TODO:
    # uniqify by uuid
    for proc in list(find_info_items(info_section, {'process', 'process_context'})):
        name = proc['properties']['name']
        if name not in {'coco_align_geotiffs', 'coco_align'}:
            print(f'name={name}')
            print(proc['properties']['start_timestamp'])
            print(proc['properties']['emissions']['run_id'])
            print('proc = {}'.format(ub.urepr(proc, nl=2)))
        # print(proc['properties']['name'])


# def trace_kwcoco_lineage(fpath):
def load_iarpa_evaluation(fpath):
    """
    Args:
        fpath (PathLike | str):
            path to the IARPA summary json file

    Returns:
        Dict: containing keys:
            metrics -
                which just contains a flat Dict[str, float] metric dictionary
            iarpa_info -
                which contains ALL of the information parsed out of the summary json file.

    Ignore:
        fpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/eval/trk/package_epoch0_step41.pt.pt/Drop4-BAS_KR_R001.kwcoco/trk_pxl_fd9e1a95/trk_poly_9f08fb8c/merged/summary2.json'
    """
    iarpa_json = _load_json(fpath)
    metrics = {}
    unique_regions = set()
    if 'best_bas_rows' in iarpa_json:
        best_bas_rows = pd.read_json(
            io.StringIO(json.dumps(iarpa_json['best_bas_rows'])),
            orient='table')
        bas_row = best_bas_rows.loc['__macro__'].reset_index().iloc[0]

        metrics.update({
            'bas_tp': bas_row['tp sites'],
            'bas_fp': bas_row['fp sites'],
            'bas_fn': bas_row['fn sites'],
            'bas_ntrue': bas_row['total sites'],
            'bas_npred': bas_row['proposed slices'],
            'bas_ppv': bas_row['precision'],
            'bas_tpr': bas_row['recall (PD)'],
            'bas_ffpa': bas_row['ffpa'],
            'bas_f1': bas_row['F1'],
            'rho': bas_row['rho'],
            'tau': bas_row['tau'],
            'bas_space_FAR': bas_row['spatial FAR'],
            'bas_time_FAR': bas_row['temporal FAR'],
            'bas_image_FAR': bas_row['images FAR'],
        })
        alpha = 1.0
        metrics['bas_faa_f1'] = metrics['bas_f1'] * (1 - metrics['bas_ffpa']) ** alpha

        unique_regions.update(
            best_bas_rows.index.levels[best_bas_rows.index.names.index('region_id')].difference({'__macro__', '__micro__'})
        )

    if 'sc_df' in iarpa_json:
        sc_json_data = iarpa_json['sc_df']
        sc_json_text = json.dumps(sc_json_data)
        try:
            sc_df = pd.read_json(io.StringIO(sc_json_text), orient='table')
        except pd.errors.IntCastingNaNError:
            # This seems like a pandas bug. It looks like it can save a Int64
            # with NaN exteions, but it can't load it back in.
            sc_json_data = iarpa_json['sc_df']
            walker = ub.IndexableWalker(sc_json_data)
            for path, val in walker:
                if path[-1] == 'extDtype' and val == 'Int64':
                    walker[path] = 'number'
                if path[-1] == 'type' and val == 'integer':
                    walker[path] = 'number'
            sc_json_text = json.dumps(sc_json_data)
            sc_df = pd.read_json(io.StringIO(sc_json_text), orient='table')

        site_prep_f1 = sc_df.loc['__macro__', 'Site Preparation']['F1']
        active_f1 = sc_df.loc['__macro__', 'Active Construction']['F1']

        site_prep_te = sc_df.loc['__macro__', 'Site Preparation']['TE']
        active_te = sc_df.loc['__macro__', 'Active Construction']['TE']
        post_te = sc_df.loc['__macro__', 'Post Construction']['TE']

        metrics.update({
            # 'mean_f1': sc_df.loc['F1'].mean(),
            'sc_macro_f1': (site_prep_f1 + active_f1) / 2,
            'macro_f1_siteprep': site_prep_f1,
            'macro_f1_active': active_f1,

            'sc_macro_te': sum([site_prep_te, active_te, post_te]) / 3,
            'macro_te_siteprep': site_prep_te,
            'macro_te_active': active_te,
            'macro_te_post': post_te,
        })

        if '__micro__' in sc_df.index:
            # Not sure why micro sometimes is not included.
            metrics['sc_micro_f1'] = sc_df.loc['__micro__']['F1'].mean()
        else:
            metrics['sc_micro_f1'] = np.nan

        unique_regions.update(
            sc_df.index.levels[sc_df.index.names.index('region_id')].difference({'__macro__', '__micro__'})
        )

    # quick and dirty way to get access to single-region results
    region_ids = ','.join(sorted(unique_regions))
    iarpa_json['region_ids'] = region_ids

    iarpa_result = {
        'metrics': metrics,
        'iarpa_json': iarpa_json,
    }
    return iarpa_result


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


def load_pxl_eval(fpath, expt_dvc_dpath=None, arg_prefix='', mode=0, with_param_types=True):
    from kwcoco.coco_evaluator import CocoSingleResult
    # from geowatch.utils import result_analysis
    # from kwutil import util_time
    measure_info = _load_json(fpath)

    # meta = measure_info['meta']
    # pred_info = meta['info']
    # dvc_dpath = expt_dvc_dpath

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
    json_info = measure_info.copy()

    extra_attrs = {}
    param_types = None

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
        'json_info': json_info,
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
            # import geowatch
            # dvc_dpath = geowatch.find_dvc_dpath()
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


global_ureg()


def _add_prefix(prefix, dict_):
    return {prefix + k: v for k, v in dict_.items()}


def relevant_pred_pxl_config(pred_pxl_config, dvc_dpath=None, arg_prefix=''):
    # TODO: better way of inferring what params are relevant
    # This should be metadata a scriptconfig object can hold.
    pred_config = {}
    pred_config['tta_fliprot'] = pred_pxl_config.get('tta_fliprot', 0)
    pred_config['tta_time'] = pred_pxl_config.get('tta_time', 0)
    pred_config['chip_overlap'] = pred_pxl_config['chip_overlap']
    pred_config['input_space_scale'] = pred_pxl_config.get('input_space_scale', None)
    pred_config['window_space_scale'] = pred_pxl_config.get('window_space_scale', None)
    pred_config['output_space_scale'] = pred_pxl_config.get('output_space_scale', None)
    pred_config['time_span'] = pred_pxl_config.get('time_span', None)
    pred_config['time_sampling'] = pred_pxl_config.get('time_sampling', None)
    pred_config['time_steps'] = pred_pxl_config.get('time_steps', None)
    pred_config['chip_dims'] = pred_pxl_config.get('chip_dims', None)
    pred_config['set_cover_algo'] = pred_pxl_config.get('set_cover_algo', None)
    pred_config['resample_invalid_frames'] = pred_pxl_config.get('resample_invalid_frames', None)
    pred_config['use_cloudmask'] = pred_pxl_config.get('use_cloudmask', None)
    package_fpath = pred_pxl_config['package_fpath']
    test_dataset = pred_pxl_config['test_dataset']
    if dvc_dpath is not None:
        package_fpath = resolve_cross_machine_path(package_fpath, dvc_dpath)
        test_dataset = resolve_cross_machine_path(test_dataset, dvc_dpath)

    # pred_config['model_fpath'] = package_fpath
    # pred_config['in_dataset'] = test_dataset
    pred_config['package_fpath'] = package_fpath
    pred_config['test_dataset'] = test_dataset

    # FIXME: use a common heuristic to transform a path into a model.
    test_dataset = ub.Path(test_dataset)
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


def parse_resource_item(item, arg_prefix='', add_prefix=True):
    resources = {}

    ureg = global_ureg()
    pred_prop = item['properties']

    start_time = util_time.coerce_datetime(pred_prop.get('start_timestamp', None))
    end_time = util_time.coerce_datetime(pred_prop.get('end_timestamp', pred_prop.get('stop_timestamp', None)))
    iters_per_second = pred_prop.get('iters_per_second', None)
    if start_time is None or end_time is None:
        total_hours = None
    else:
        total_hours = (end_time - start_time).total_seconds() / (60 * 60)
    resources['total_hours'] = total_hours
    if iters_per_second is not None:
        resources['iters_per_second'] = iters_per_second

    if 'duration' in pred_prop:
        resources['duration'] = pred_prop['duration']

    try:
        vram = pred_prop['device_info']['allocated_vram']
        vram_gb = ureg.parse_expression(f'{vram} bytes').to('gigabytes').m
        resources['vram_gb'] = vram_gb
    except KeyError:
        ...

    hardware_parts = []

    if 'machine' in pred_prop:
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
    if add_prefix:
        resources = _add_prefix(arg_prefix + 'resource.', resources)
    return resources


def find_pred_pxl_item(pred_info):
    pred_items = list(find_info_items(
        pred_info,
        {'process', 'process_context'},
        'geowatch.tasks.fusion.predict'
    ))
    assert len(pred_items) == 1
    pred_item = pred_items[0]
    return pred_item


def find_info_items(info, query_type, query_name=None):
    from kwutil import util_pattern
    if query_name is None:
        query_name = '*'
    query_name_pattern = util_pattern.MultiPattern.coerce(query_name)
    query_type_pattern = util_pattern.MultiPattern.coerce(query_type)
    for item in info:
        if query_type_pattern.match(item['type']):
            name = item['properties']['name']
            if query_name_pattern.match(name):
                yield item


@ub.memoize
def _load_json(fpath):
    # memo hack for development
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data


def find_track_item(tracker_info):
    tracker_alias = {
        'geowatch.cli.run_tracker',
        'geowatch.cli.run_tracker',
    }
    track_items = list(find_info_items(
        tracker_info,
        {'process', 'process_context'},
        tracker_alias
    ))
    if len(track_items) != 1:
        raise AssertionError(ub.paragraph(
            f'''
            We should be able to find exactly 1 tracker process item,
            but instead we found {len(track_items)}
            '''))
        ...
    track_item = track_items[0]
    return track_item


def find_metrics_framework_item(info):
    task_aliases = {
        'geowatch.cli.run_metrics_framework',
    }
    items = list(find_info_items(
        info,
        {'process', 'process_context'},
        task_aliases
    ))
    if len(items) != 1:
        raise AssertionError(ub.paragraph(
            f'''
            We should be able to find exactly 1 tracker process item,
            but instead we found {len(items)}
            '''))
        ...
    item = items[0]
    return item


def find_pxl_eval_item(info):
    task_aliases = {
        'geowatch.tasks.fusion.evaluate',
    }
    items = list(find_info_items(
        info,
        {'process', 'process_context'},
        task_aliases
    ))
    if len(items) != 1:
        raise AssertionError(ub.paragraph(
            f'''
            We should be able to find exactly 1 tracker process item,
            but instead we found {len(items)}
            '''))
        ...
    item = items[0]
    return item


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

r"""
Helper for scheduling a set of prediction + evaluation jobs

TODO:
    - [ ] Differentiate between pixel models for different tasks.
    - [ ] Allow the output of tracking to feed into activity classification



Example:

    python -m watch.mlops.schedule_evaluation \
        --params="
            matrix:
                trk.pxl.model:
                    - ./my_bas_model.pt
                trk.pxl.data.test_dataset:
                    - ./my_test_dataset/bas_ready_data.kwcoco.json
                trk.pxl.data.window_scale_space: 15GSD
                trk.pxl.data.time_sampling:
                    - "auto"
                trk.pxl.data.input_scale_space:
                    - "15GSD"
                crop.src:
                    - ./my_test_dataset/sc_query_data.kwcoco.json
                crop.regions:
                    - trk.poly.output
                act.pxl.data.test_dataset:
                    - crop.dst
                act.pxl.data.window_scale_space:
                    - auto
                act.poly.thresh:
                    - 0.1
                act.poly.use_viterbi:
                    - 0
                act.pxl.model:
                    - my_sc_model.pt
        " \
        --expt_dvc_dpath=./my_expt_dir \
        --data_dvc_dpath=./my_data_dir \
        --dynamic_skips=0 \
        --enable_pred_trk_pxl=1 \
        --enable_pred_trk_poly=1 \
        --enable_eval_trk_pxl=0 \
        --enable_eval_trk_poly=0 \
        --enable_crop=1 \
        --enable_pred_act_pxl=1 \
        --enable_pred_act_poly=1 \
        --enable_eval_act_pxl=0 \
        --enable_eval_act_poly=0 \
        --enable_viz_pred_trk_poly=0 \
        --enable_viz_pred_act_poly=0 \
        --enable_links=0 \
        --devices="0,1" --queue_size=2 \
        --backend=serial --skip_existing=0 \
        --run=0
"""
import ubelt as ub
import shlex
import json
import scriptconfig as scfg
# import watch
import shlex  # NOQA
import json  # NOQA
from watch.utils.lightning_ext import util_globals  # NOQA
import kwarray  # NOQA
import cmd_queue
import networkx as nx
import itertools as it
from watch.utils.util_param_grid import expand_param_grid, dotdict_to_nested


class ScheduleEvaluationConfig(scfg.Config):
    """
    Builds commands and optionally schedules them.
    """
    default = {
        'devices': scfg.Value('auto', help='if using tmux or serial, indicate which gpus are available for use as a comma separated list: e.g. 0,1'),
        'run': scfg.Value(False, help='if False, only prints the commands, otherwise executes them'),
        'virtualenv_cmd': scfg.Value(None, help='command to activate a virtualenv if needed. (might have issues with slurm backend)'),
        'skip_existing': scfg.Value(False, help='if True dont submit commands where the expected products already exist'),
        'backend': scfg.Value('tmux', help='can be tmux, slurm, or maybe serial in the future'),
        'queue_name': scfg.Value('schedule-eval', help='Name of the queue'),

        'pred_workers': scfg.Value(4, help='number of prediction workers in each process'),

        'shuffle_jobs': scfg.Value(True, help='if True, shuffles the jobs so they are submitted in a random order'),
        'annotations_dpath': scfg.Value(None, help='path to IARPA annotations dpath for IARPA eval'),

        'expt_dvc_dpath': None,
        'data_dvc_dpath': None,

        'check_other_sessions': scfg.Value('auto', help='if True, will ask to kill other sessions that might exist'),
        'queue_size': scfg.Value('auto', help='if auto, defaults to number of GPUs'),

        # These enabled flags should probably be pushed off to params
        'enable_pred_trk_pxl': scfg.Value(True, isflag=True, help='BAS heatmap'),
        'enable_pred_trk_poly': scfg.Value(True, isflag=True, help='BAS tracking'),
        'enable_crop': scfg.Value(True, isflag=True, help='SC tracking'),
        'enable_pred_act_pxl': scfg.Value(True, isflag=True, help='SC heatmaps'),
        'enable_pred_act_poly': scfg.Value(True, isflag=True, help='SC tracking'),
        'enable_viz_pred_act_poly': scfg.Value(False, isflag=True, help='if true draw predicted tracks for SC'),

        'enable_eval_trk_pxl': scfg.Value(True, isflag=True, help='BAS heatmap evaluation'),
        'enable_eval_trk_poly': scfg.Value(True, isflag=True, help='BAS tracking evaluation'),
        'enable_eval_act_pxl': scfg.Value(True, isflag=True, help='SC heatmaps evaluation'),
        'enable_eval_act_poly': scfg.Value(True, isflag=True, help='SC tracking evaluation'),
        'enable_viz_pred_trk_poly': scfg.Value(False, isflag=True, help='if true draw predicted tracks for BAS'),
        'enable_links': scfg.Value(True, isflag=True, help='if true enable symlink jobs'),

        'dynamic_skips': scfg.Value(True, isflag=True, help='if true, each a test is appened to each job to skip itself if its output exists'),

        'draw_heatmaps': scfg.Value(1, isflag=True, help='if true draw heatmaps on eval'),
        'draw_curves': scfg.Value(1, isflag=True, help='if true draw curves on eval'),

        'partition': scfg.Value(None, help='specify slurm partition (slurm backend only)'),
        'mem': scfg.Value(None, help='specify slurm memory per task (slurm backend only)'),

        'params': scfg.Value(None, type=str, help='a yaml/json grid/matrix of prediction params'),
    }


def schedule_evaluation(cmdline=False, **kwargs):
    r"""
    First ensure that models have been copied to the DVC repo in the
    appropriate path. (as noted by model_dpath)

    Ignore:
        from watch.mlops.schedule_evaluation import *  # NOQA
        kwargs = {'params': ub.codeblock(
            '''
            - matrix:
                ###
                ### BAS Pixel Prediction
                ###
                trk.pxl.model:
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=73-step=37888.pt.pt
                trk.pxl.data.test_dataset:
                    - ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_bas.json
                trk.pxl.data.tta_time: 0
                trk.pxl.data.chip_overlap: 0.3
                trk.pxl.data.window_scale_space: 10GSD
                trk.pxl.data.input_scale_space: 15GSD
                trk.pxl.data.output_scale_space: 15GSD
                trk.pxl.data.time_span: auto
                trk.pxl.data.time_sampling: auto
                trk.pxl.data.time_steps: auto
                trk.pxl.data.chip_dims: auto
                trk.pxl.data.set_cover_algo: None
                trk.pxl.data.resample_invalid_frames: 1
                trk.pxl.data.use_cloudmask: 1
                ###
                ### BAS Polygon Prediction
                ###
                trk.poly.thresh: [0.10, 0.15]
                trk.poly.morph_kernel: 3
                trk.poly.norm_ord: 1
                trk.poly.agg_fn: probs
                trk.poly.thresh_hysteresis: None
                trk.poly.moving_window_size: None
                trk.poly.polygon_fn: heatmaps_to_polys
                ###
                ### SC Pixel Prediction
                ###
                crop.src: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc.json
                crop.regions:
                    - trk.poly.output
                    - truth
                act.pxl.data.test_dataset:
                    - crop.dst
                    # - ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_sc.json
                act.pxl.model:
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
            ''')}
        cmdline = 0
    """
    import watch
    config = ScheduleEvaluationConfig(cmdline=cmdline, data=kwargs)
    print('ScheduleEvaluationConfig config = {}'.format(ub.repr2(dict(config), nl=1, si=1)))

    expt_dvc_dpath = config['expt_dvc_dpath']
    data_dvc_dpath = config['data_dvc_dpath']
    if expt_dvc_dpath is None:
        expt_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_expt')
    if data_dvc_dpath is None:
        data_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_data')
    data_dvc_dpath = ub.Path(data_dvc_dpath)
    expt_dvc_dpath = ub.Path(expt_dvc_dpath)

    from watch.mlops.expt_manager import ExperimentState
    # start using the experiment state logic as the path and metadata
    # organization logic
    state = ExperimentState(expt_dvc_dpath, '*')

    # Get truth annotations
    annotations_dpath = config['annotations_dpath']
    if annotations_dpath is None:
        annotations_dpath = data_dvc_dpath / 'annotations'
    annotations_dpath = ub.Path(annotations_dpath)
    region_model_dpath = annotations_dpath / 'region_models'

    # Expand paramater search grid
    # arg = config['params']
    all_param_grid = expand_param_grid(config['params'])

    grid_item_defaults = ub.udict({
        'trk.pxl.model': None,
        'trk.pxl.data.test_dataset': None,
        'trk.poly.thresh': 0.1,

        'crop.src': None,
        'crop.context_factor': 1.5,
        'crop.regions': 'truth',

        'act.pxl.model': None,
        'act.pxl.data.test_dataset': None,
        'act.poly.thresh': 0.1,
    })

    resolved_rows = []
    # Resolve parameters for each row
    for item in ub.ProgIter(all_param_grid, desc='resolving row'):
        row = resolve_pipeline_row(grid_item_defaults, state,
                                   region_model_dpath, expt_dvc_dpath, item)
        resolved_rows.append(row)

    # from rich import print

    queue_dpath = expt_dvc_dpath / '_cmd_queue_schedule'
    queue_dpath.ensuredir()

    devices = config['devices']
    print('devices = {!r}'.format(devices))
    if devices == 'auto':
        GPUS = _auto_gpus()
    else:
        GPUS = None if devices is None else ensure_iterable(devices)
    print('GPUS = {!r}'.format(GPUS))

    queue_size = config['queue_size']
    if queue_size == 'auto':
        queue_size = len(GPUS)

    common_submitkw = dict(
        partition=config['partition'],
        mem=config['mem']
    )
    environ = {}
    queue = cmd_queue.Queue.create(
        config['backend'], name=config['queue_name'],
        size=queue_size, environ=environ,
        dpath=queue_dpath, gres=GPUS
    )

    virtualenv_cmd = config['virtualenv_cmd']
    if virtualenv_cmd:
        queue.add_header_command(virtualenv_cmd)

    # TODO: parameterize
    perf_params = {
        'trk.pxl.batch_size': 1,
        'trk.pxl.workers': 4,
        'trk.pxl.devices': '0,',
        'trk.pxl.accelerator': 'gpu',

        'act.pxl.batch_size': 1,
        'act.pxl.workers': 4,
        'act.pxl.devices': '0,',
        'act.pxl.accelerator': 'gpu',
    }

    # Each row represents a single source-to-sink pipeline run, but multiple
    # rows may share pipeline steps. This is handled by having unique ids per
    # job that depend on their outputs.
    for row in ub.ProgIter(resolved_rows, desc='submiting pipelines'):
        paths = row['paths']
        paths['true_annotations_dpath'] = annotations_dpath
        paths['true_site_dpath'] = paths['true_annotations_dpath'] / 'site_models'
        paths['true_region_dpath'] = paths['true_annotations_dpath'] / 'region_models'
        task_params = row['task_params']
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())

        trk_pxl_params = task_params['trk.pxl']
        trk_poly_params = task_params['trk.poly']
        crop_params = task_params['crop']
        act_pxl_params = task_params['act.pxl']
        act_poly_params = task_params['act.poly']
        condensed = row['condensed']

        ### define the dag of this row item.

        steps = {}

        # TODO: detect if BAS team features are needed and compute those here.
        # Do we update the dataset paths here or beforehand?

        step = Pipeline.pred_trk_pxl(perf_params, trk_pxl_params, **paths)
        steps[step.name] = step

        step = Pipeline.pred_trk_poly(trk_poly_params, **paths)
        steps[step.name] = step

        step = Pipeline.act_crop(crop_params, **paths)
        steps[step.name] = step

        # TODO: detect if SC team features are needed and compute those here.

        step = Pipeline.pred_act_pxl(perf_params, act_pxl_params, **paths)
        steps[step.name] = step

        step = Pipeline.pred_act_poly(act_poly_params, **paths)
        steps[step.name] = step

        step = Pipeline.eval_trk_pxl(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.eval_trk_poly(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.viz_pred_trk_poly(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.eval_act_pxl(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.eval_act_poly(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.viz_pred_act_poly(condensed, **paths)
        steps[step.name] = step

        for step in steps.values():
            step.enabled = config['enable_' + step.name]
            step.otf_cache = config['dynamic_skips']

        ADD_LINKS = config['enable_links']
        if ADD_LINKS:
            # Make jobs to symlink things on the fly
            linkable_pairs = [
                {'pred': 'pred_trk_pxl_dpath', 'eval': 'eval_trk_pxl_dpath'},
                {'pred': 'pred_act_pxl_dpath', 'eval': 'eval_act_pxl_dpath'},
                {'pred': 'pred_trk_poly_dpath', 'eval': 'eval_trk_poly_dpath'},
                {'pred': 'pred_act_poly_dpath', 'eval': 'eval_act_poly_dpath'},
            ]
            for pair in linkable_pairs:
                pred_key = pair['pred']
                eval_key = pair['eval']
                pred_dpath = paths[pred_key]
                eval_dpath = paths[eval_key]
                pred_lpath = eval_dpath / '_pred_link'
                eval_lpath = pred_dpath / '_eval_link'
                out_paths = {
                    pred_key.replace('dpath', 'lpath'): pred_lpath,
                    eval_key.replace('dpath', 'lpath'): eval_lpath,
                }
                name = 'link_' + eval_key.replace('eval_', '')
                parts = [
                    f'ln -sf "{eval_dpath}" "{eval_lpath}"',
                    f'ln -sf "{pred_dpath}" "{pred_lpath}"',
                ]
                command = ' && '.join(parts)
                step = Step(name, command,
                            in_paths=paths & {pred_key, eval_key},
                            out_paths=out_paths,
                            resources={'cpus': 1})
                steps[step.name] = step
                step.enabled = True
                step.otf_cache = False

            # The track to act is multiway.
            # One way pair
            if condensed['regions_id'] != 'truth':
                key1 = 'pred_trk_poly_dpath'
                key2 = 'crop_dpath'
                dpath1 = paths[key1]
                dpath2 = paths[key2]
                lpath_parent = dpath2 / '_crop_links'
                lpath1 = lpath_parent / condensed['crop_id']
                lpath2 = dpath1 / '_pred_trk_poly_link'
                eval_lpath = pred_dpath / '_eval_link'
                out_paths = {
                    key1.replace('dpath', 'lpath'): lpath1,
                    key2.replace('dpath', 'lpath'): lpath2,
                }
                name = 'link_trk_crop'
                parts = [
                    f'mkdir {lpath_parent}',
                    f'ln -sf "{dpath1}" "{lpath1}"',
                    f'ln -sf "{dpath2}" "{lpath2}"',
                ]
                command = ' && '.join(parts)
                step = Step(name, command,
                            in_paths=paths & {pred_key, eval_key},
                            out_paths=out_paths,
                            resources={'cpus': 1})
                steps[step.name] = step
                step.enabled = True
                step.otf_cache = True

        skip_existing = config['skip_existing']
        g = Pipeline.connect_steps(steps, skip_existing)

        #
        # Submit steps to the scheduling queue
        for node in g.graph['order']:
            # Skip duplicate jobs
            step = g.nodes[node]['step']
            if step.node_id in queue.named_jobs:
                continue
            depends = []
            for other, _ in list(g.in_edges(node)):
                dep_step = g.nodes[other]['step']
                if dep_step.enabled:
                    depends.append(dep_step.node_id)

            if step.will_exist and step.enabled:
                queue.submit(command=step.command, name=step.node_id,
                             depends=depends, **common_submitkw)

    print('queue = {!r}'.format(queue))
    # print(f'{len(queue)=}')
    with_status = 0
    with_rich = 0
    # queue.write_network_text()
    queue.rprint(with_status=with_status, with_rich=with_rich)

    for job in queue.jobs:
        # TODO: should be able to set this as a queue param.
        job.log = False

    # RUN
    if config['run']:
        # ub.cmd('bash ' + str(driver_fpath), verbose=3, check=True)
        queue.run(block=True, check_other_sessions=config['check_other_sessions'])
    else:
        driver_fpath = queue.write()
        print('Wrote script: to run execute:\n{}'.format(driver_fpath))


def ensure_iterable(inputs):
    return inputs if ub.iterable(inputs) else [inputs]


def _auto_gpus():
    # TODO: liberate the needed code from netharn
    # Use all unused devices
    import netharn as nh
    GPUS = []
    for gpu_idx, gpu_info in nh.device.gpu_info().items():
        print('gpu_idx = {!r}'.format(gpu_idx))
        print('gpu_info = {!r}'.format(gpu_info))
        if len(gpu_info['procs']) == 0:
            GPUS.append(gpu_idx)
    return GPUS


def resolve_pipeline_row(grid_item_defaults, state, region_model_dpath, expt_dvc_dpath, item):
    """
    The idea is to handle any dependencies between parameters, extract any
    path-based metadata, and construct every path that this pipeline will
    touch.
    """
    from watch.mlops.expt_manager import ExperimentState
    state = ExperimentState(expt_dvc_dpath, '*')
    item = grid_item_defaults | item
    # print('item = {}'.format(ub.repr2(item, nl=1)))
    nested = dotdict_to_nested(item)

    condensed = {}
    paths = {}

    # Might not need this exactly
    try:
        pkg_trk_pixel_pathcfg = state._parse_pattern_attrs(
            state.templates['pkg_trk_pxl_fpath'], item['trk.pxl.model'])
    except RuntimeError:
        ...  # user specified a custom model
        condensed['dataset_code'] = 'dset_code_unknown'
        condensed['trk_expt'] = 'trk_expt_unknown'
        condensed['trk_model'] = ub.Path(item['trk.pxl.model']).name
    else:
        # fixme: dataset code is ambiguous between BAS and SC
        # pkg_trk_pixel_pathcfg.pop('dataset_code', None)
        condensed.update(pkg_trk_pixel_pathcfg)

    condensed['expt_dvc_dpath'] = expt_dvc_dpath

    ### BAS / TRACKING ###

    trk_pxl  = nested['trk']['pxl']
    trk_poly = nested['trk']['poly']
    trk_pxl_params = ub.udict(trk_pxl['data']) - {'test_dataset'}
    trk_poly_params = ub.udict(trk_poly)

    condensed['trk_model'] = state._condense_model(item['trk.pxl.model'])
    # TODO:
    # based on the model, we should infer if we need team features or not.

    condensed['test_trk_dset'] = state._condense_test_dset(item['trk.pxl.data.test_dataset'])
    condensed['trk_pxl_cfg'] = state._condense_cfg(trk_pxl_params, 'trk_pxl')
    condensed['trk_poly_cfg'] = state._condense_cfg(trk_poly_params, 'trk_poly')

    paths['pkg_trk_pxl_fpath'] = ub.Path(item['trk.pxl.model'])
    paths['trk_test_dataset_fpath'] = item['trk.pxl.data.test_dataset']
    paths['pred_trk_pxl_dpath'] = ub.Path(state.templates['pred_trk_pxl_dpath'].format(**condensed))
    paths['pred_trk_pxl_fpath'] = ub.Path(state.templates['pred_trk_pxl_fpath'].format(**condensed))
    paths['eval_trk_pxl_dpath'] = ub.Path(state.templates['eval_trk_pxl_dpath'].format(**condensed))
    paths['eval_trk_pxl_fpath'] = ub.Path(state.templates['eval_trk_pxl_fpath'].format(**condensed))

    paths['pred_trk_poly_sites_fpath'] = ub.Path(state.templates['pred_trk_poly_sites_fpath'].format(**condensed))
    paths['pred_trk_poly_site_summaries_fpath'] = ub.Path(state.templates['pred_trk_poly_site_summaries_fpath'].format(**condensed))
    paths['pred_trk_poly_sites_dpath'] = ub.Path(state.templates['pred_trk_poly_sites_dpath'].format(**condensed))
    paths['pred_trk_poly_site_summaries_dpath'] = ub.Path(state.templates['pred_trk_poly_site_summaries_dpath'].format(**condensed))

    paths['pred_trk_poly_viz_stamp'] = ub.Path(state.templates['pred_trk_poly_viz_stamp'].format(**condensed))

    paths['pred_trk_poly_dpath'] = ub.Path(state.templates['pred_trk_poly_dpath'].format(**condensed))
    paths['pred_trk_poly_kwcoco'] = ub.Path(state.templates['pred_trk_poly_kwcoco'].format(**condensed))

    paths['eval_trk_poly_fpath'] = ub.Path(state.templates['eval_trk_poly_fpath'].format(**condensed))
    paths['eval_trk_poly_dpath'] = ub.Path(state.templates['eval_trk_poly_dpath'].format(**condensed))

    trackid_deps = {}
    trackid_deps = ub.udict(condensed) & state.hashid_dependencies['trk_poly_id']
    condensed['trk_poly_id'] = state._condense_cfg(trackid_deps, 'trk_poly_id')

    ### CROPPING ###

    crop_params = ub.udict(nested['crop']).copy()
    crop_src_fpath = crop_params.pop('src')
    paths['crop_src_fpath'] = crop_src_fpath
    if crop_params['regions'] == 'truth':
        # Crop job depends only on true annotations
        paths['crop_regions'] = str(region_model_dpath) + '/*.geojson'
        condensed['regions_id'] = 'truth'  # todo: version info
        site_summary = str(region_model_dpath) + '/*.geojson'
    if crop_params['regions'] == 'trk.poly.output':
        # Crop job depends on track predictions
        paths['crop_regions'] = paths['pred_trk_poly_site_summaries_fpath']
        condensed['regions_id'] = condensed['trk_poly_id']
        site_summary = state.templates['pred_trk_poly_site_summaries_fpath'].format(**condensed)

    crop_params['regions'] = paths['crop_regions']
    condensed['crop_cfg'] = state._condense_cfg(crop_params, 'crop')
    condensed['crop_src_dset'] = state._condense_test_dset(crop_src_fpath)

    crop_id_deps = ub.udict(condensed) & state.hashid_dependencies['crop_id']
    condensed['crop_id'] = state._condense_cfg(crop_id_deps, 'crop_id')
    paths['crop_dpath'] = ub.Path(state.templates['crop_dpath'].format(**condensed))
    paths['crop_fpath'] = ub.Path(state.templates['crop_fpath'].format(**condensed))
    condensed['crop_dst_dset'] = state._condense_test_dset(paths['crop_fpath'])

    ### SC / ACTIVITY ###
    act_pxl  = nested['act']['pxl']
    act_poly = nested['act']['poly']
    act_pxl_params = ub.udict(act_pxl['data']) - {'test_dataset'}
    act_poly_params = ub.udict(act_poly)
    act_poly_params['site_summary'] = site_summary

    condensed['act_model'] = state._condense_model(item['act.pxl.model'])
    # TODO:
    # based on the model, we should infer if we need team features or not.
    condensed['act_pxl_cfg'] = state._condense_cfg(act_pxl_params, 'act_pxl')
    condensed['act_poly_cfg'] = state._condense_cfg(act_poly_params, 'act_poly')

    try:
        pkg_act_pixel_pathcfg = state._parse_pattern_attrs(state.templates['pkg_act_pxl_fpath'], item['act.pxl.model'])
    except RuntimeError:
        ...  # user specified a custom model
        condensed['dataset_code'] = 'dset_code_unknown'
        condensed['act_expt'] = 'act_expt_unknown'
        condensed['act_model'] = ub.Path(item['act.pxl.model']).name
    else:
        condensed.update(pkg_act_pixel_pathcfg)

    # paths['act_test_dataset_fpath'] = item['act.pxl.data.test_dataset']
    if item['act.pxl.data.test_dataset'] == 'crop.dst':
        # Activity prediction depends on a cropping job
        paths['act_test_dataset_fpath'] = paths['crop_fpath']
    else:
        # Activity prediction has no dependencies in this case.
        paths['act_test_dataset_fpath'] = item['act.pxl.data.test_dataset']
    condensed['test_act_dset'] = state._condense_test_dset(paths['act_test_dataset_fpath'])

    paths['pkg_act_pxl_fpath'] = ub.Path(item['act.pxl.model'])

    paths['pred_act_poly_sites_fpath'] = ub.Path(state.templates['pred_act_poly_sites_fpath'].format(**condensed))
    # paths['pred_act_poly_site_summaries_fpath'] = ub.Path(state.templates['pred_act_poly_site_summaries_fpath'].format(**condensed))
    paths['pred_act_poly_sites_dpath'] = ub.Path(state.templates['pred_act_poly_sites_dpath'].format(**condensed))
    # paths['pred_act_poly_site_summaries_dpath'] = ub.Path(state.templates['pred_act_poly_site_summaries_dpath'].format(**condensed))

    paths['pred_act_pxl_dpath'] = ub.Path(state.templates['pred_act_pxl_dpath'].format(**condensed))
    paths['pred_act_poly_dpath'] = ub.Path(state.templates['pred_act_poly_dpath'].format(**condensed))
    paths['pred_act_pxl_fpath'] = ub.Path(state.templates['pred_act_pxl_fpath'].format(**condensed))
    paths['eval_act_pxl_fpath'] = ub.Path(state.templates['eval_act_pxl_fpath'].format(**condensed))
    paths['eval_act_pxl_dpath'] = ub.Path(state.templates['eval_act_pxl_dpath'].format(**condensed))
    paths['eval_act_poly_fpath'] = ub.Path(state.templates['eval_act_poly_fpath'].format(**condensed))
    paths['eval_act_poly_dpath'] = ub.Path(state.templates['eval_act_poly_dpath'].format(**condensed))
    paths['pred_act_poly_kwcoco'] = ub.Path(state.templates['pred_act_poly_kwcoco'].format(**condensed))
    paths['pred_act_poly_viz_stamp'] = ub.Path(state.templates['pred_act_poly_viz_stamp'].format(**condensed))

    # paths['eval_act_tmp_dpath'] = pred_act_row['eval_act_poly_dpath'] / '_tmp'

    task_params = {
        'trk.pxl': trk_pxl_params,
        'trk.poly': trk_poly_params,
        'crop': crop_params,
        'act.pxl': act_pxl_params,
        'act.poly': act_poly_params,
    }

    row = {
        'condensed': condensed,
        'paths': paths,
        'item': item,
        'task_params': task_params,
    }
    return row


def resolve_package_paths(model_globstr, expt_dvc_dpath):
    import rich
    # import glob
    from watch.utils import util_pattern

    # HACK FOR DVC PTH FIXME:
    # if str(model_globstr).endswith('.txt'):
    #     from watch.utils.simple_dvc import SimpleDVC
    #     print('model_globstr = {!r}'.format(model_globstr))
    #     # if expt_dvc_dpath is None:
    #     #     expt_dvc_dpath = SimpleDVC.find_root(ub.Path(model_globstr))

    def expand_model_list_file(model_lists_fpath, expt_dvc_dpath=None):
        """
        Given a file containing paths to models, expand it into individual
        paths.
        """
        expanded_fpaths = []
        lines = [line for line in ub.Path(model_globstr).read_text().split('\n') if line]
        missing = []
        for line in lines:
            if expt_dvc_dpath is not None:
                package_fpath = ub.Path(expt_dvc_dpath / line)
            else:
                package_fpath = ub.Path(line)
            if package_fpath.is_file():
                expanded_fpaths.append(package_fpath)
            else:
                missing.append(line)
        if missing:
            rich.print('[yellow] WARNING: missing = {}'.format(ub.repr2(missing, nl=1)))
            rich.print(f'[yellow] WARNING: specified a models-of-interest.txt and {len(missing)} / {len(lines)} models were missing')
        return expanded_fpaths

    print('model_globstr = {!r}'.format(model_globstr))
    model_globstr = util_pattern.MultiPattern.coerce(model_globstr)
    package_fpaths = []
    # for package_fpath in glob.glob(model_globstr, recursive=True):
    for package_fpath in model_globstr.paths(recursive=True):
        package_fpath = ub.Path(package_fpath)
        if package_fpath.name.endswith('.txt'):
            # HACK FOR PATH OF MODELS
            model_lists_fpath = package_fpath
            expanded_fpaths = expand_model_list_file(model_lists_fpath, expt_dvc_dpath=expt_dvc_dpath)
            package_fpaths.extend(expanded_fpaths)
        else:
            package_fpaths.append(package_fpath)

    if len(package_fpaths) == 0:
        import pathlib
        if '*' not in str(model_globstr):
            package_fpaths = [ub.Path(model_globstr)]
        elif isinstance(model_globstr, (str, pathlib.Path)):
            # Warn the user if they gave a bad model globstr (this is just one
            # of the many potential ways things could go wrong)
            glob_path = ub.Path(model_globstr)
            def _concrete_glob_part(path):
                " Find the resolved part of the glob path "
                concrete_parts = []
                for p in path.parts:
                    if '*' in p:
                        break
                    concrete_parts.append(p)
                return ub.Path(*concrete_parts)
            concrete = _concrete_glob_part(glob_path)
            if not concrete.exists():
                rich.print('[yellow] WARNING: part of the model_globstr does not exist: {}'.format(concrete))

    return package_fpaths


class Step:
    def __init__(step, name, command, in_paths, out_paths, resources):
        step.name = name
        step._command = command
        step.in_paths = ub.udict(in_paths)
        step.out_paths = ub.udict(out_paths)
        step.resources = ub.udict(resources)
        #
        # Set later
        step.enabled = None
        step.will_exist = None
        step.otf_cache = True  # on-the-fly cache checking

    @property
    def node_id(step):
        """
        The experiment manager constructs output paths such that they
        are unique given the specific set of inputs and parameters. Thus
        the output paths are sufficient to determine a unique id per step.
        """
        return step.name + '_' + ub.hash_data(step.out_paths)[0:12]

    @property
    def command(step):
        if step.otf_cache and step.enabled != 'redo':
            return step.test_is_computed_command() + ' || ' + step._command
        else:
            return step._command

    def test_is_computed_command(step):
        test_expr = ' -a '.join(
            [f'-e "{p}"' for p in step.out_paths.values()])
        test_cmd = 'test ' +  test_expr
        return test_cmd

    @ub.memoize_property
    def does_exist(self):
        return all(self.out_paths.map_values(lambda p: p.exists()).values())


class Pipeline:
    """
    Registers how to call each step in the pipeline
    """

    @classmethod
    def connect_steps(cls, steps, skip_existing=False):
        """
        Build the graph that represents this pipeline
        """
        # Determine the interaction / dependencies between step inputs /
        # outputs
        g = nx.DiGraph()
        outputs_to_step = ub.ddict(list)
        inputs_to_step = ub.ddict(list)
        for step in steps.values():
            for path in step.out_paths.values():
                outputs_to_step[path].append(step.name)
            for path in step.in_paths.values():
                inputs_to_step[path].append(step.name)
            g.add_node(step.name, step=step)

        inputs_to_step = ub.udict(inputs_to_step)
        outputs_to_step = ub.udict(outputs_to_step)

        common = list((inputs_to_step & outputs_to_step).keys())
        for path in common:
            isteps = inputs_to_step[path]
            osteps = outputs_to_step[path]
            for istep, ostep in it.product(isteps, osteps):
                g.add_edge(ostep, istep)

        #
        # Determine which steps are enabled / disabled
        sorted_nodes = list(nx.topological_sort(g))
        g.graph['order'] = sorted_nodes

        for node in sorted_nodes:
            step = g.nodes[node]['step']
            # if config['skip_existing']:
            ancestors_will_exist = all(
                g.nodes[ancestor]['step'].will_exist
                for ancestor in nx.ancestors(g, step.name)
            )
            if skip_existing and step.enabled != 'redo' and step.does_exist:
                step.enabled = False
            step.will_exist = (
                (step.enabled and ancestors_will_exist) or
                step.does_exist
            )

        if 0:
            from cmd_queue.util.util_networkx import write_network_text
            write_network_text(g)
        return g

    def __init__(pipe):
        pass

    @staticmethod
    def _make_argstr(params):
        parts = [f'    --{k}={v} \\' for k, v in params.items()]
        return chr(10).join(parts).lstrip().rstrip('\\')

    def act_crop(crop_params, **paths):
        paths = ub.udict(paths)
        from watch.cli import coco_align
        confobj = coco_align.__config__
        known_args = set(confobj.default.keys())
        assert not len(ub.udict(crop_params) - known_args), 'unknown args'
        crop_params = {
            'geo_preprop': 'auto',
            'keep': 'img',
            'force_nodata': -9999,
            'rpc_align_method': 'orthorectify',
            'target_gsd': 4,
            'site_summary': True,
        } | ub.udict(crop_params)

        # The best setting of this depends on if the data is remote or not.
        # When networking, around 20+ workers is a good idea, but that's a very
        # bad idea for local images or if the images are too big.
        # Parametarizing would be best.
        CROP_IMAGE_WORKERS = 16
        CROP_AUX_WORKERS = 8

        perf_options = {
            'verbose': 1,
            'workers': CROP_IMAGE_WORKERS,
            'aux_workers': CROP_AUX_WORKERS,
            'debug_valid_regions': False,
            'visualize': False,
        }
        crop_kwargs = { **paths }
        crop_kwargs['crop_params_argstr'] = Pipeline._make_argstr(crop_params)
        crop_kwargs['crop_perf_argstr'] = Pipeline._make_argstr(perf_options)

        command = ub.codeblock(
            r'''
            python -m watch.cli.coco_align \
                --src "{crop_src_fpath}" \
                --dst "{crop_fpath}" \
                {crop_params_argstr} \
                {crop_perf_argstr} \
            ''').format(**crop_kwargs).strip().rstrip('\\')

        # FIXME: parametarize and only if we need secrets
        # secret_fpath = ub.Path('$HOME/code/watch/secrets/secrets').expand()
        # # if ub.Path.home().name.startswith('jon'):
        #     # if secret_fpath.exists():
        #     #     secret_fpath
        #         # command = f'source {secret_fpath} && ' + command
        command = 'AWS_DEFAULT_PROFILE=iarpa GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR ' + command

        name = 'crop'
        step = Step(name, command,
                    in_paths=paths & {'crop_regions', 'crop_src_fpath'},
                    out_paths=paths & {'crop_fpath', 'crop_dpath'},
                    resources={'cpus': 2})
        return step

    def pred_trk_pxl(perf_params, trk_pxl_params, **paths):
        paths = ub.udict(paths)
        workers = perf_params['trk.pxl.workers']
        perf_options = {
            'num_workers': workers,
            'devices': perf_params['trk.pxl.devices'],
            'accelerator': perf_params['trk.pxl.accelerator'],
            'batch_size': perf_params['trk.pxl.batch_size'],
        }
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        pred_trk_pxl_kw =  { **paths }
        pred_trk_pxl_kw['params_argstr'] = Pipeline._make_argstr(trk_pxl_params)
        pred_trk_pxl_kw['perf_argstr'] = Pipeline._make_argstr(perf_options)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.predict \
                --package_fpath={pkg_trk_pxl_fpath} \
                --test_dataset={trk_test_dataset_fpath} \
                --pred_dataset={pred_trk_pxl_fpath} \
                {params_argstr} \
                {perf_argstr}
            ''').format(**pred_trk_pxl_kw)
        name = 'pred_trk_pxl'
        step = Step(name, command,
                    in_paths=paths & {'pkg_trk_pxl_fpath',
                                      'trk_test_dataset_fpath'},
                    out_paths=paths & {'pred_trk_pxl_fpath'},
                    resources={'cpus': workers, 'gpus': 2})
        return step

    def pred_act_pxl(perf_params, act_pxl_params, **paths):
        """
        Idea:
            Step(<command_template>, <path-templates>)

            path_templates=[
                Template('pkg_act_pxl_fpath', 'packages/{act_expt}/{act_model}.pt')
                Template('act_test_dataset_fpath', '???'),
                Template('pred_act_pxl_fpath', '{pred_act_pxl_dpath}/pred.kwcoco.json'),
            ]
        """
        paths = ub.udict(paths)
        workers = perf_params['act.pxl.workers']
        perf_options = {
            'num_workers': workers,  # note: the "num_workers" cli arg should be changed to "workers" everywhere
            'devices': perf_params['act.pxl.devices'],
            'accelerator': perf_params['act.pxl.accelerator'],
            'batch_size': perf_params['act.pxl.batch_size'],
        }
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        pred_act_pxl_kw =  { **paths }
        pred_act_pxl_kw['params_argstr'] = Pipeline._make_argstr(act_pxl_params)
        pred_act_pxl_kw['perf_argstr'] = Pipeline._make_argstr(perf_options)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.predict \
                --package_fpath={pkg_act_pxl_fpath} \
                --test_dataset={act_test_dataset_fpath} \
                --pred_dataset={pred_act_pxl_fpath} \
                {params_argstr} \
                {perf_argstr}
            ''').format(**pred_act_pxl_kw)
        name = 'pred_act_pxl'
        step = Step(name, command,
                    in_paths=paths & {
                        'pkg_act_pxl_fpath',
                        'act_test_dataset_fpath'
                    },
                    out_paths=paths & {'pred_act_pxl_fpath'},
                    resources={'cpus': workers, 'gpus': 2})
        return step

    def pred_trk_poly(trk_poly_params, **paths):
        paths = ub.udict(paths)
        pred_trk_poly_kw = { **paths }
        cfg = trk_poly_params.copy()
        if 'thresh_hysteresis' in cfg:
            if isinstance(cfg['thresh_hysteresis'], str):
                cfg['thresh_hysteresis'] = util_globals.restricted_eval(
                    cfg['thresh_hysteresis'].format(**cfg))

        if 'moving_window_size' in cfg:
            if isinstance(cfg['moving_window_size'], str):
                cfg['moving_window_size'] = util_globals.restricted_eval(
                    cfg['moving_window_size'].format(**cfg))
        else:
            cfg['moving_window_size'] = None

        if cfg['moving_window_size'] is None:
            cfg['polygon_fn'] = 'heatmaps_to_polys'
        else:
            cfg['polygon_fn'] = 'heatmaps_to_polys_moving_window'
        # kwargs['params_argstr'] = Pipeline(trk_poly_params)
        pred_trk_poly_kw['track_kwargs_str'] = shlex.quote(json.dumps(cfg))

        command = ub.codeblock(
            r'''
            python -m watch.cli.run_tracker \
                "{pred_trk_pxl_fpath}" \
                --default_track_fn saliency_heatmaps \
                --track_kwargs {track_kwargs_str} \
                --clear_annots \
                --out_sites_dir "{pred_trk_poly_sites_dpath}" \
                --out_site_summaries_dir "{pred_trk_poly_site_summaries_dpath}" \
                --out_sites_fpath "{pred_trk_poly_sites_fpath}" \
                --out_site_summaries_fpath "{pred_trk_poly_site_summaries_fpath}" \
                --out_kwcoco "{pred_trk_poly_kwcoco}"
            ''').format(**pred_trk_poly_kw)
        name = 'pred_trk_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_pxl_fpath'},
                    out_paths=paths & {
                        'pred_trk_poly_dpath',
                        'pred_trk_poly_sites_fpath',
                        'pred_trk_poly_site_summaries_fpath',
                        'pred_trk_poly_kwcoco'},
                    resources={'cpus': 2})
        return step

    def pred_act_poly(act_poly_params, **paths):
        paths = ub.udict(paths)
        pred_act_poly_kw = paths.copy()
        # pred_act_row['site_summary_glob'] = (region_model_dpath / '*.geojson')
        # pred_act_poly_kw['site_summary_glob'] = (pred_act_poly_kw['region_model_dpath'] / '*.geojson')
        # pred_act_poly_kw['site_summary_glob'] = (paths['pred_trk_poly_sites_dpath'] / '*.geojson')
        pred_act_poly_kw['site_summary'] = act_poly_params.pop('site_summary')
        actclf_cfg = {
            'boundaries_as': 'polys',
        }
        actclf_cfg.update(act_poly_params)
        pred_act_poly_kw['kwargs_str'] = shlex.quote(json.dumps(actclf_cfg))
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_tracker \
                "{pred_act_pxl_fpath}" \
                --default_track_fn class_heatmaps \
                --track_kwargs {kwargs_str} \
                --site_summary '{site_summary}' \
                --out_sites_fpath "{pred_act_poly_sites_fpath}" \
                --out_sites_dir "{pred_act_poly_sites_dpath}" \
                --out_kwcoco "{pred_act_poly_kwcoco}"
            ''').format(**pred_act_poly_kw)
        name = 'pred_act_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_act_pxl_fpath'},
                    out_paths=paths & {'pred_act_poly_dpath',
                                       'pred_act_poly_sites_fpath',
                                       'pred_act_poly_kwcoco'},
                    resources={'cpus': 2})
        return step

    def eval_trk_pxl(condensed, **paths):
        paths = ub.udict(paths)
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        eval_trk_pxl_kw = { **paths }
        extra_opts = {
            'draw_curves': True,  # todo: parametarize
            'draw_heatmaps': True,  # todo: parametarize
            'viz_thresh': 0.2,
            'workers': 2,
        }
        eval_trk_pxl_kw['extra_argstr'] = Pipeline._make_argstr(extra_opts)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.evaluate \
                --true_dataset={trk_test_dataset_fpath} \
                --pred_dataset={pred_trk_pxl_fpath} \
                --eval_dpath={eval_trk_pxl_dpath} \
                --score_space=video \
                {extra_argstr}
            ''').format(**eval_trk_pxl_kw)
        name = 'eval_trk_pxl'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_pxl_fpath'},
                    out_paths=paths & {'eval_trk_pxl_fpath',
                                       'eval_trk_pxl_dpath'},
                    resources={'cpus': 2})
        return step

    def eval_act_pxl(condensed, **paths):
        paths = ub.udict(paths)
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        eval_act_pxl_kw = { **paths }
        extra_opts = {
            'draw_curves': True,
            'draw_heatmaps': True,
            'viz_thresh': 0.2,
            'workers': 2,
        }
        eval_act_pxl_kw['extra_argstr'] = Pipeline._make_argstr(extra_opts)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.evaluate \
                --true_dataset={act_test_dataset_fpath} \
                --pred_dataset={pred_act_pxl_fpath} \
                --eval_dpath={eval_act_pxl_dpath} \
                --score_space=video \
                {extra_argstr}
            ''').format(**eval_act_pxl_kw).strip().rstrip('\\')
        name = 'eval_act_pxl'
        step = Step(name, command,
                    in_paths=paths & {'pred_act_pxl_fpath'},
                    out_paths=paths & {'eval_act_pxl_fpath',
                                       'eval_act_pxl_dpath'},
                    resources={'cpus': 2})
        return step

    def viz_pred_trk_poly(condensed, **paths):
        paths = ub.udict(paths)
        viz_pred_trk_poly_kw = paths.copy()
        viz_pred_trk_poly_kw['extra_header'] = f"\\n{condensed['trk_pxl_cfg']}-{condensed['trk_poly_cfg']}"
        viz_pred_trk_poly_kw['viz_channels'] = "red|green|blue,salient"
        command = ub.codeblock(
            r'''
            smartwatch visualize \
                "{pred_trk_poly_kwcoco}" \
                --channels="{viz_channels}" \
                --stack=only \
                --workers=avail/2 \
                --workers=avail/2 \
                --extra_header="{extra_header}" \
                --animate=True && touch {pred_trk_poly_viz_stamp}
            ''').format(**viz_pred_trk_poly_kw)
        name = 'viz_pred_trk_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_poly_kwcoco'},
                    out_paths=paths & {'pred_trk_poly_viz_stamp'},
                    resources={'cpus': 2})
        return step

    def viz_pred_act_poly(condensed, **paths):
        paths = ub.udict(paths)
        viz_pred_act_poly_kw = paths.copy()
        viz_pred_act_poly_kw['extra_header'] = f"\\n{condensed['act_pxl_cfg']}-{condensed['act_poly_cfg']}"
        # viz_pred_act_poly_kw['viz_channels'] = 'red|green|blue,No Activity|Site Preparation|Active Construction|Post Construction'
        viz_pred_act_poly_kw['viz_channels'] = 'red|green|blue,No Activity|Active Construction|Post Construction'
        command = ub.codeblock(
            r'''
            smartwatch visualize \
                "{pred_act_poly_kwcoco}" \
                --channels="{viz_channels}" \
                --stack=only \
                --workers=avail/2 \
                --extra_header="{extra_header}" \
                --animate=True && touch {pred_act_poly_viz_stamp}
            ''').format(**viz_pred_act_poly_kw)
        name = 'viz_pred_act_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_act_poly_kwcoco'},
                    out_paths=paths & {'pred_act_poly_viz_stamp'},
                    resources={'cpus': 2})
        return step

    def eval_trk_poly(condensed, **paths):
        paths = ub.udict(paths)
        eval_trk_poly_kw = { **paths }
        eval_trk_poly_kw['eval_trk_poly_dpath'] = eval_trk_poly_kw['eval_trk_poly_dpath']
        eval_trk_poly_kw['eval_trk_poly_tmp_dpath'] = eval_trk_poly_kw['eval_trk_poly_dpath'] / '_tmp'
        eval_trk_poly_kw['name_suffix'] = '-'.join([
            condensed['trk_model'],
            condensed['trk_pxl_cfg'],
            condensed['trk_poly_cfg'],
        ])
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_metrics_framework \
                --merge=True \
                --name "{name_suffix}" \
                --true_site_dpath "{true_site_dpath}" \
                --true_region_dpath "{true_region_dpath}" \
                --pred_sites "{pred_trk_poly_sites_fpath}" \
                --tmp_dir "{eval_trk_poly_tmp_dpath}" \
                --out_dir "{eval_trk_poly_dpath}" \
                --merge_fpath "{eval_trk_poly_fpath}"
            ''').format(**eval_trk_poly_kw)
        name = 'eval_trk_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_poly_sites_fpath'},
                    out_paths=paths & {'eval_trk_poly_dpath',
                                       'eval_trk_poly_fpath'},
                    resources={'cpus': 2})
        return step

    def eval_act_poly(condensed, **paths):
        """
        Idea:
            Step(<command_template>, <path-templates>)

            path_templates=[
                Template('true_site_dpath', 'true_site_dpath')
                Template('pred_act_poly_sites_fpath', '{pred_act_poly_dpath}/activity_tracks.kwcoco.json'),
            ]
        """
        paths = ub.udict(paths)
        eval_act_poly_kw = paths.copy()
        eval_act_poly_kw['name_suffix'] = '-'.join([
            # condensed['crop_dst_dset']
            condensed['test_act_dset'],
            condensed['act_model'],
            condensed['act_pxl_cfg'],
            condensed['act_poly_cfg'],
        ])
        eval_act_poly_kw['eval_act_poly_dpath'] = eval_act_poly_kw['eval_act_poly_dpath']
        eval_act_poly_kw['eval_act_poly_tmp_dpath'] = eval_act_poly_kw['eval_act_poly_dpath'] / '_tmp'
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_metrics_framework \
                --merge=True \
                --name "{name_suffix}" \
                --true_site_dpath "{true_site_dpath}" \
                --true_region_dpath "{true_region_dpath}" \
                --pred_sites "{pred_act_poly_sites_fpath}" \
                --tmp_dir "{eval_act_poly_tmp_dpath}" \
                --out_dir "{eval_act_poly_dpath}" \
                --merge_fpath "{eval_act_poly_fpath}" \
            ''').format(**eval_act_poly_kw)
        name = 'eval_act_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_act_poly_sites_fpath'},
                    out_paths=paths & {'eval_act_poly_fpath',
                                       'eval_act_poly_dpath'},
                    resources={'cpus': 2})
        return step


__notes__ = """
If this is going to be a real mlops framework, then we need to abstract the
pipeline. The user needs to define what the steps are, but then they need to
explicitly connect them. We can't make the assumptions we are currently using.

Ignore:

    # We can use our CLIs as definitions of the pipeline as long as the config
    # object has enough metadata. With scriptconfig+jsonargparse we should be
    # able to do this.

    import watch.cli.run_metrics_framework
    import watch.cli.run_tracker
    import watch.tasks.fusion.predict

    watch.cli.run_tracker.__config__.__default__

    list(watch.tasks.fusion.predict.make_predict_config().__dict__.keys())


    from watch.tasks.tracking.from_heatmap import NewTrackFunction
    from watch.tasks.tracking.from_heatmap import TimeAggregatedBAS
    from watch.tasks.tracking.from_heatmap import TimeAggregatedSC
    from watch.tasks.tracking.from_heatmap import TimeAggregatedHybrid

    import jsonargparse
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(TimeAggregatedBAS, nested_key='trk.poly')
    parser.add_class_arguments(TimeAggregatedSC, nested_key='act.poly')
    # parser.add_subclass_arguments(NewTrackFunction, nested_key='poly')
    parser.print_help()

    parser.parse_known_args([])


    import jsonargparse
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--foo')
    args = parser.parse_args(args=[])
    parser.save(args)

"""


if __name__ == '__main__':
    schedule_evaluation(cmdline=True)

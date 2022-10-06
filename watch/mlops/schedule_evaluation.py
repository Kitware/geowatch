"""
Helper for scheduling a set of prediction + evaluation jobs

TODO:
    - [ ] Differentiate between pixel models for different tasks.
    - [ ] Allow the output of tracking to feed into activity classification
"""
import ubelt as ub
import scriptconfig as scfg


class ScheduleEvaluationConfig(scfg.Config):
    """
    Builds commands and optionally schedules them.
    """
    default = {
        # 'model_globstr': scfg.Value(None, help='one or more glob patterns that match the models to predict/evaluate on'),
        # 'trk_model_globstr': scfg.Value(None, help='one or more glob patterns that match the models to predict/evaluate on'),
        # 'act_model_globstr': scfg.Value(None, help='one or more glob patterns that match the models to predict/evaluate on'),
        # 'trk_test_dataset': scfg.Value(None, help='path to the test dataset to predict/evaluate on for BAS'),
        # 'act_test_dataset': scfg.Value(None, help='path to the test dataset to predict/evaluate/crop from for SC'),

        'devices': scfg.Value('auto', help='if using tmux or serial, indicate which gpus are available for use as a comma separated list: e.g. 0,1'),
        'run': scfg.Value(False, help='if False, only prints the commands, otherwise executes them'),
        'virtualenv_cmd': scfg.Value(None, help='command to activate a virtualenv if needed. (might have issues with slurm backend)'),
        'skip_existing': scfg.Value(False, help='if True dont submit commands where the expected products already exist'),
        'backend': scfg.Value('tmux', help='can be tmux, slurm, or maybe serial in the future'),

        'pred_workers': scfg.Value(4, help='number of prediction workers in each process'),

        'sidecar2': scfg.Value(True, help='if True uses parallel sidecar pattern, otherwise nested'),
        'shuffle_jobs': scfg.Value(True, help='if True, shuffles the jobs so they are submitted in a random order'),
        'annotations_dpath': scfg.Value(None, help='path to IARPA annotations dpath for IARPA eval'),

        'expt_dvc_dpath': None,
        'data_dvc_dpath': None,

        'check_other_sessions': scfg.Value('auto', help='if True, will ask to kill other sessions that might exist'),
        'queue_size': scfg.Value('auto', help='if auto, defaults to number of GPUs'),

        'enable_pred_pxl': scfg.Value(True, isflag=True, help='if False, then prediction is not run', alias=['enable_pred']),
        'enable_eval_pxl': scfg.Value(True, isflag=True, help='if False, then evaluation is not run', alias=['enable_eval']),
        'enable_pred_trk': scfg.Value(False, isflag=True, help='if True, enable tracking', alias=['enable_track']),
        'enable_eval_trk': scfg.Value(False, isflag=True, help='if True, enable iapra BAS evalaution', alias=['enable_iarpa_eval']),
        'enable_pred_act': scfg.Value(False, isflag=True, help='if True, enable actclf', alias=['enable_actclf']),
        'enable_eval_act': scfg.Value(False, isflag=True, help='if True, enable iapra SC evalaution', alias=['enable_actclf_eval']),

        'enable_pred_trk_viz': scfg.Value(False, isflag=True, help='if true draw predicted tracks'),

        'draw_heatmaps': scfg.Value(1, isflag=True, help='if true draw heatmaps on eval'),
        'draw_curves': scfg.Value(1, isflag=True, help='if true draw curves on eval'),

        'partition': scfg.Value(None, help='specify slurm partition (slurm backend only)'),
        'mem': scfg.Value(None, help='specify slurm memory per task (slurm backend only)'),

        'params': scfg.Value(None, type=str, help='a yaml/json grid/matrix of prediction params'),
    }


class Task(dict):
    def __init__(self, *args, manager=None, skip_existing=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = manager
        self.skip_existing = skip_existing

    @property
    def name(self):
        return self['name']

    def should_compute_task(task_info):
        # Check if each dependency will exist by the time we run this job
        deps_will_exist = []
        for req in task_info['requires']:
            deps_will_exist.append(task_info.manager[req]['will_exist'])
        all_deps_will_exist = all(deps_will_exist)
        task_info['all_deps_will_exist'] = all_deps_will_exist

        output = task_info.get('output', None)

        if not all_deps_will_exist:
            # If dependencies wont exist, then we cant run
            enabled = False
        else:
            # If we can run, then do it this task is requested
            if task_info.skip_existing and (output and output.exists()):
                enabled = task_info['recompute']
            else:
                enabled = bool(task_info['requested'])
        # Only enable the task if we requested it and its dependencies will
        # exist.
        task_info['enabled'] = enabled
        # Mark if we do exist, or we will exist
        will_exist = enabled or (output and output.exists())
        task_info['will_exist'] = will_exist
        return task_info['enabled']

    def prefix_command(task_info, command):
        """
        Augments the command so it is lazy if its output exists

        TODO: incorporate into cmdq
        """
        if task_info['recompute']:
            stamp_fpath = task_info['output']
            if stamp_fpath is not None:
                command = f'test -f "{stamp_fpath}" || \\\n  ' + command
        return command


def build_crop_job():
    crop_config = {
        'src': None,
        'dst': None,
        'region_globstr': None,
        'force_nodata': -9999,  #
        'align_keep': 'img',
        'rpc_align_method': 'orthorectify',
        'verbose': 1,
        'target_gsd': 4,
        'align_workers': 2,
        'align_aux_workers': 4,
        'debug_valid_regions': 0,
        'debug_valid_regions': 0,
        'include_channels': None,
        'exclude_channels': None,
    }
    command = ub.codeblock(
        rf'''
        python -m watch.cli.coco_align_geotiffs \
            --src "{crop_config['src']}" \
            --dst "{crop_config['dst']}" \
            --regions "{crop_config['region_globstr']}" \
            --context_factor=1.5 \
            --geo_preprop=auto \
            --keep={crop_config['align_keep']} \
            --force_nodata={crop_config['force_nodata']} \
            --include_channels="{crop_config['include_channels']}" \
            --exclude_channels="{crop_config['exclude_channels']}" \
            --visualize=False \
            --debug_valid_regions={crop_config['debug_valid_regions']} \
            --rpc_align_method {crop_config['rpc_align_method']} \
            --verbose={crop_config['verbose']} \
            --target_gsd={crop_config['target_gsd']} \
            --workers={crop_config['align_workers']}
            --aux_workers={crop_config['align_aux_workers']} \
        ''')



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
                    act.crop.src: ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/kwcoco_for_sc.json
                    act.crop.regions:
                        - trk.poly.output
                        - truth
                    act.pxl.data.test_dataset:
                        - act.pxl.crop.dst
                        - ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_sc.json
                    act.pxl.model:
                        - ~/data/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
            ''')}
        cmdline = 0
    """
    import watch
    import shlex
    import json
    from watch.utils.lightning_ext import util_globals
    import kwarray
    import cmd_queue

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
    all_param_grid = handle_param_grid(config['params'])

    grid_item_defaults = ub.udict({
        'trk.pxl.model': None,
        'trk.pxl.data.test_dataset': None,
        'trk.poly.thresh': 0.1,

        'act.crop.src': None,
        'act.crop.regions': 'truth',

        'act.pxl.model': None,
        'act.pxl.data.test_dataset': None,
        'act.poly.thresh': 0.1,
    })

    def build_trk_pxl_job(trk_pxl):
        package_fpath = trk_pxl['model']
        trk_pxl['condensed'] = condensed
        package_info = {}
        package_info['package_fpath'] = package_fpath
        package_info['condensed'] = condensed
        pass

    resolved_rows = []
    # Resolve parameters for each row
    for item in all_param_grid:

        state = ExperimentState(expt_dvc_dpath, '*')
        item = grid_item_defaults | item
        nested = dotdict_to_nested(item)

        condensed = {}
        paths = {}

        # Might not need this exactly
        pkg_trk_pixel_pathcfg = state._parse_pattern_attrs(
            state.templates['pkg_trk_pxl_fpath'], item['trk.pxl.model'])
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
        condensed['test_trk_dset'] = state._condense_test_dset(item['trk.pxl.data.test_dataset'])
        condensed['trk_pxl_cfg'] = state._condense_cfg(trk_pxl_params, 'trk_pxl')
        condensed['trk_poly_cfg'] = state._condense_cfg(trk_poly_params, 'trk_poly')

        paths['trk_model_fpath'] = ub.Path(item['trk.pxl.model'])
        paths['trk_test_dataset_fpath'] = item['trk.pxl.data.test_dataset']
        paths['pred_trk_pxl_fpath'] = ub.Path(state.templates['pred_trk_pxl_fpath'].format(**condensed))
        paths['eval_trk_pxl_dpath'] = ub.Path(state.templates['eval_trk_pxl_dpath'].format(**condensed))
        paths['eval_trk_pxl_fpath'] = ub.Path(state.templates['eval_trk_pxl_fpath'].format(**condensed))
        paths['pred_trk_poly_fpath'] = ub.Path(state.templates['pred_trk_poly_fpath'].format(**condensed))
        paths['eval_trk_poly_fpath'] = ub.Path(state.templates['eval_trk_poly_fpath'].format(**condensed))

        trackid_deps = {}
        trackid_deps = ub.udict(condensed) & (
            {'trk_model', 'test_trk_dset', 'trk_pxl_cfg', 'trk_poly_cfg'})
        condensed['trk_poly_id'] = state._condense_cfg(trackid_deps, 'trk_poly_id')

        ### CROPPING ###

        crop_params = ub.udict(nested['act']['crop']).copy()
        crop_src_fpath = crop_params.pop('src')
        if crop_params['regions'] == 'truth':
            # Crop job depends only on true annotations
            crop_params['regions'] = str(region_model_dpath) + '/*.geojson'
            condensed['regions_id'] = 'truth'  # todo: version info
        if crop_params['regions'] == 'trk.poly.output':
            # Crop job depends on track predictions
            crop_params['regions'] = state.templates['pred_trk_poly_fpath'].format(**condensed)
            condensed['regions_id'] = condensed['trk_poly_id']
        condensed['crop_cfg'] = state._condense_cfg(crop_params, 'crop')
        condensed['crop_src_dset'] = state._condense_test_dset(crop_src_fpath)

        crop_id_deps = ub.udict(condensed) & (
            {'regions_id', 'crop_cfg', 'crop_src_dset'})
        condensed['crop_id'] = state._condense_cfg(crop_id_deps, 'crop_id')
        paths['crop_dpath'] = ub.Path(state.templates['crop_dpath'].format(**condensed))
        paths['crop_fpath'] = ub.Path(state.templates['crop_fpath'].format(**condensed))
        condensed['crop_dst_dset'] = state._condense_test_dset(paths['crop_fpath'])

        ### SC / ACTIVITY ###
        act_pxl  = nested['act']['pxl']
        act_poly = nested['act']['poly']
        act_pxl_params = ub.udict(act_pxl['data']) - {'test_dataset'}
        act_poly_params = ub.udict(act_poly)

        condensed['act_model'] = state._condense_model(item['act.pxl.model'])
        condensed['act_pxl_cfg'] = state._condense_cfg(act_pxl_params, 'act_pxl')
        condensed['act_poly_cfg'] = state._condense_cfg(act_poly_params, 'act_poly')

        pkg_act_pixel_pathcfg = state._parse_pattern_attrs(state.templates['pkg_act_pxl_fpath'], item['act.pxl.model'])
        # pkg_act_pixel_pathcfg.pop('dataset_code', None)
        condensed.update(pkg_act_pixel_pathcfg)

        # paths['act_test_dataset_fpath'] = item['act.pxl.data.test_dataset']
        if item['act.pxl.data.test_dataset'] == 'act.pxl.crop.dst':
            # Activity prediction depends on a cropping job
            paths['act_test_dataset_fpath'] = paths['crop_fpath']
        else:
            # Activity prediction has no dependencies in this case.
            paths['act_test_dataset_fpath'] = item['act.pxl.data.test_dataset']
        condensed['test_act_dset'] = state._condense_test_dset(paths['act_test_dataset_fpath'])

        paths['act_model_fpath'] = ub.Path(item['act.pxl.model'])
        paths['pred_act_poly_fpath'] = ub.Path(state.templates['pred_act_poly_fpath'].format(**condensed))
        paths['pred_act_pxl_fpath'] = ub.Path(state.templates['pred_act_pxl_fpath'].format(**condensed))
        paths['eval_act_pxl_fpath'] = ub.Path(state.templates['eval_act_pxl_fpath'].format(**condensed))
        paths['eval_act_pxl_dpath'] = ub.Path(state.templates['eval_act_pxl_dpath'].format(**condensed))
        paths['eval_act_poly_fpath'] = ub.Path(state.templates['eval_act_poly_fpath'].format(**condensed))
        paths['eval_act_poly_dpath'] = ub.Path(state.templates['eval_act_poly_dpath'].format(**condensed))

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
        resolved_rows.append(row)

    # model_globstr = config['model_globstr']
    # trk_dataset_fpath = config['trk_test_dataset']
    # act_dataset_fpath = config['act_test_dataset']

    draw_curves = config['draw_curves']
    draw_heatmaps = config['draw_heatmaps']

    # if model_globstr is None and trk_dataset_fpath is None:
    #     raise ValueError('model_globstr and test_dataset are required')

    # # Gather the appropriate requested models
    # if model_globstr is not None:
    #     package_fpaths = resolve_package_paths(model_globstr, expt_dvc_dpath)
    # task_model_fpaths = {
    #     'trk_pxl': [],
    #     'act_pxl': [],
    # }
    # if config['trk_model_globstr'] is not None:
    #     task_model_fpaths['trk_pxl'] = resolve_package_paths(config['trk_model_globstr'], expt_dvc_dpath)

    # if config['act_model_globstr'] is not None:
    #     task_model_fpaths['act_pxl'] = resolve_package_paths(config['act_model_globstr'], expt_dvc_dpath)

    # package_fpaths = resolve_package_paths(model_globstr, expt_dvc_dpath)

    print(f'expt_dvc_dpath={expt_dvc_dpath}')
    print(f'data_dvc_dpath={data_dvc_dpath}')

    skip_existing = config['skip_existing']
    with_pred = config['enable_pred_pxl']  # TODO: allow caching
    with_eval = config['enable_eval_pxl']
    with_track = config['enable_pred_trk']
    with_iarpa_eval = config['enable_eval_trk']
    recompute = False
    def check_recompute(flag, depends_flags=[]):
        return recompute or flag == 'redo' or any(f == 'redo' for f in depends_flags)
    recompute_pred = check_recompute(with_pred)
    recompute_eval = check_recompute(with_pred, [with_pred])
    recompute_track = check_recompute(with_track, [with_pred])
    recompute_iarpa_eval = check_recompute(with_iarpa_eval, [with_pred, recompute_track])

    workers_per_queue = config['pred_workers']

    trk_dataset_fpath = ub.Path(trk_dataset_fpath)
    if not trk_dataset_fpath.exists():
        print('warning test dataset does not exist')

    task_pkg_rows = {
        'trk_pxl': [],
        'act_pxl': [],
    }
    for task, package_fpaths in task_model_fpaths.items():
        candidate_pkg_rows = []
        for package_fpath in package_fpaths:
            condensed = state._parse_pattern_attrs(state.templates['pkg_' + task], package_fpath)
            # Overwrite expt_dvc_dpath because it was parsed as a src dir,
            # but we are going to use it as a dst dir
            condensed['expt_dvc_dpath'] = expt_dvc_dpath
            package_info = {}
            package_info['package_fpath'] = package_fpath
            package_info['condensed'] = condensed
            candidate_pkg_rows.append(package_info)
        task_pkg_rows[task] = candidate_pkg_rows

    for candidate_pkg_rows in task_pkg_rows['trk_pxl']:
        print('candidate_pkg_rows = {}'.format(ub.repr2(candidate_pkg_rows, nl=1)))
        pass

    print('task_pkg_rows = {}'.format(ub.repr2(task_pkg_rows, nl=1)))

    # Build the info we need to submit every prediction job of interest
    candidate_pred_rows = []
    trk_test_dset = state._condense_test_dset(trk_dataset_fpath)
    act_test_dset = state._condense_test_dset(act_dataset_fpath)
    for pkg_row in ub.ProgIter(candidate_pkg_rows, desc='build pred rows'):
        for pred_cfg in pred_pxl_param_grid:
            pred_pxl_row = pkg_row.copy()
            condensed  = pred_pxl_row['condensed'].copy()

            condensed['trk_test_dset'] = trk_test_dset
            condensed['act_test_dset'] = act_test_dset
            condensed['pred_cfg'] = state._condense_pred_cfg(pred_cfg)

            pred_pxl_row['condensed'] = condensed
            pred_pxl_row['pred_cfg'] = pred_cfg
            pred_pxl_row['trk_dataset_fpath'] = trk_dataset_fpath
            # TODO: make using these templates easier
            pred_pxl_row['pred_pxl_fpath'] = ub.Path(state.templates['pred_trk_pxl'].format(**condensed))
            pred_pxl_row['eval_pxl_fpath'] = ub.Path(state.templates['eval_trk_pxl'].format(**condensed))
            pred_pxl_row['eval_pxl_dpath'] = pred_pxl_row['eval_pxl_fpath'].parent.parent

            # TODO: Technically the there should be an evaluation config list
            # that is looped over for every prediction, but for now they are
            # 1-to-1. A general ML-ops framwork should provide this.

            # TODO: these are really part of the pred_cfg, even though they are
            # semi-non-impactful, handle them gracefully
            # These are part of the pxl eval config
            pred_pxl_row['draw_curves'] = draw_curves
            pred_pxl_row['draw_heatmaps'] = draw_heatmaps
            candidate_pred_rows.append(pred_pxl_row)

    # FIXME: reintroduce
    # if with_eval == 'redo':
    #     # Need to dvc unprotect
    #     # TODO: this can be a job in the queue
    #     needs_unprotect = []
    #     for pred_pxl_row in candidate_pred_rows:
    #         eval_metrics_fpath = ub.Path(pred_pxl_row['eval_pxl_fpath'])
    #         eval_metrics_dvc_fpath = eval_metrics_fpath.augment(tail='.dvc')
    #         if eval_metrics_dvc_fpath.exists():
    #             needs_unprotect.append(eval_metrics_fpath)
    #     if needs_unprotect:
    #         # TODO: use the dvc experiment manager for this.
    #         # This should not be our concern
    #         from watch.utils.simple_dvc import SimpleDVC
    #         simple_dvc = SimpleDVC(expt_dvc_dpath)
    #         simple_dvc.unprotect(needs_unprotect)

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

    environ = {}
    queue = cmd_queue.Queue.create(config['backend'], name='schedule-eval',
                                   size=queue_size, environ=environ,
                                   dpath=queue_dpath, gres=GPUS)

    virtualenv_cmd = config['virtualenv_cmd']
    if virtualenv_cmd:
        queue.add_header_command(virtualenv_cmd)

    common_submitkw = dict(
        partition=config['partition'],
        mem=config['mem']
    )

    if config['shuffle_jobs']:
        candidate_pred_rows = kwarray.shuffle(candidate_pred_rows)

    for pred_pxl_row in ub.ProgIter(candidate_pred_rows, desc='build track rows'):
        package_fpath = pred_pxl_row['package_fpath']
        pred_cfg = pred_pxl_row['pred_cfg']
        condensed = pred_pxl_row['condensed']
        pred_pxl_row['name_suffix'] = '-'.join([
            condensed['model'],
            condensed['pred_cfg'],
        ])

        if config['shuffle_jobs']:
            pred_trk_param_grid = kwarray.shuffle(pred_trk_param_grid)

        # First compute children track, activity rows (todo: refactor to do
        # ealier)
        candidate_trk_rows = []
        for trk_cfg in pred_trk_param_grid:
            pred_trk_row = pred_pxl_row.copy()
            pred_trk_row['condensed'] = condensed = pred_trk_row['condensed'].copy()
            condensed['trk_cfg'] = state._condense_trk_cfg(trk_cfg)
            pred_trk_row['name_suffix'] = '-'.join([
                condensed['model'],
                condensed['pred_cfg'],
                condensed['trk_cfg'],
            ])

            pred_trk_row['pred_trk_fpath'] = ub.Path(state.templates['pred_trk'].format(**condensed))
            pred_trk_row['pred_trk_kwcoco_fpath'] = ub.Path(state.templates['pred_trk_kwcoco'].format(**condensed))
            pred_trk_row['pred_trk_dpath'] = pred_trk_row['pred_trk_fpath'].parent / 'tracked_sites'
            pred_trk_row['pred_trk_viz_stamp'] = ub.Path(state.templates['pred_trk_viz_stamp'].format(**condensed))

            pred_trk_row['eval_trk_out_fpath'] = ub.Path(state.templates['eval_trk'].format(**condensed))
            pred_trk_row['eval_trk_fpath'] = ub.Path(state.templates['eval_trk'].format(**condensed))
            pred_trk_row['eval_trk_score_dpath'] = pred_trk_row['eval_trk_fpath'].parent.parent
            pred_trk_row['eval_trk_dpath'] = pred_trk_row['eval_trk_score_dpath'].parent

            pred_trk_row['eval_trk_tmp_dpath'] = pred_trk_row['eval_trk_dpath'] / '_tmp'
            pred_trk_row['true_site_dpath'] = annotations_dpath / 'site_models'
            pred_trk_row['true_region_dpath'] = annotations_dpath / 'region_models'

            # This is the requested config, not the resolved config.
            # TODO: # differentiate.
            pred_trk_row['trk_cfg'] = trk_cfg
            candidate_trk_rows.append(pred_trk_row)

        if config['shuffle_jobs']:
            pred_act_param_grid = kwarray.shuffle(pred_act_param_grid)

        # TODO: refactor to depend on a non-truth set of predicted sites.
        candidate_act_rows = []
        for act_cfg in pred_act_param_grid:
            pred_act_row = pred_pxl_row.copy()
            pred_act_row['condensed'] = condensed = pred_act_row['condensed'].copy()
            condensed['act_cfg'] = state._condense_act_cfg(act_cfg)
            pred_act_row['pred_act_fpath'] = ub.Path(state.templates['pred_act'].format(**condensed))
            pred_trk_row['pred_act_kwcoco_fpath'] = ub.Path(state.templates['pred_act_kwcoco'].format(**condensed))
            pred_act_row['pred_act_dpath'] = pred_act_row['pred_act_fpath'].parent / 'classified_sites'

            pred_act_row['eval_act_fpath'] = ub.Path(state.templates['eval_act'].format(**condensed))
            pred_act_row['eval_act_score_dpath'] = pred_act_row['eval_act_fpath'].parent.parent
            pred_act_row['eval_act_dpath'] = pred_act_row['eval_act_score_dpath'].parent
            pred_act_row['site_summary_glob'] = (region_model_dpath / '*.geojson')

            pred_act_row['true_site_dpath'] = annotations_dpath / 'site_models'
            pred_act_row['true_region_dpath'] = annotations_dpath / 'region_models'
            pred_act_row['eval_act_tmp_dpath'] = pred_act_row['eval_act_dpath'] / '_tmp'
            pred_act_row['act_cfg'] = act_cfg

            pred_act_row['name_suffix'] = '-'.join([
                condensed['model'],
                condensed['pred_cfg'],
                condensed['act_cfg'],
            ])
            candidate_act_rows.append(pred_act_row)

        # Really should make this a class
        manager = {}
        manager['pred_pxl'] = Task(**{
            'name': 'pred_pxl',
            'requested': with_pred,
            'output': pred_pxl_row['pred_pxl_fpath'],
            'requires': [],
            'recompute': recompute_pred,
        }, manager=manager, skip_existing=skip_existing)

        manager['eval_pxl'] = Task(**{
            'name': 'eval_pxl',
            'requested': with_eval,
            'output': pred_pxl_row['eval_pxl_fpath'],
            'requires': ['pred_pxl'],
            'recompute': recompute_eval,
        }, manager=manager, skip_existing=skip_existing)

        pred_job = None
        task_info = manager['pred_pxl']
        if task_info.should_compute_task():
            predictkw = {
                'workers_per_queue': workers_per_queue,
                **pred_cfg,
                **pred_pxl_row,
            }
            predictkw['pred_cfg_argstr'] = chr(10).join(
                [f'    --{k}={v} \\' for k, v in pred_cfg.items()]).lstrip()
            command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.predict \
                    --package_fpath={package_fpath} \
                    --pred_dataset={pred_pxl_fpath} \
                    --test_dataset={trk_dataset_fpath} \
                    --num_workers={workers_per_queue} \
                    {pred_cfg_argstr}
                    --devices=0, \
                    --accelerator=gpu \
                    --batch_size=1
                ''').format(**predictkw)
            command = task_info.prefix_command(command)
            name = task_info['name'] + pred_pxl_row['name_suffix']
            pred_cpus = workers_per_queue
            pred_job = queue.submit(command, gpus=1, name=name,
                                    cpus=pred_cpus, **common_submitkw)

        task_info = manager['eval_pxl']
        if task_info.should_compute_task():

            eval_pxl_row = pred_pxl_row  # hack, for now these are 1-to-1

            command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.evaluate \
                    --true_dataset={trk_dataset_fpath} \
                    --pred_dataset={pred_pxl_fpath} \
                      --eval_dpath={eval_pxl_dpath} \
                      --score_space=video \
                      --draw_curves={draw_curves} \
                      --draw_heatmaps={draw_heatmaps} \
                      --viz_thresh=0.2 \
                      --workers=2
                ''').format(**eval_pxl_row)
            command = task_info.prefix_command(command)
            name = task_info['name'] + pred_pxl_row['name_suffix']
            task_info['job'] = queue.submit(
                command, depends=pred_job, name=name, cpus=2,
                **common_submitkw)

        for pred_trk_row in candidate_trk_rows:
            manager['pred_trk'] = Task(**{
                'name': 'pred_trk',
                'requested': config['enable_pred_trk'],
                'output': pred_trk_row['pred_trk_fpath'],
                'requires': ['pred_pxl'],
                'recompute': recompute_track,
            }, manager=manager, skip_existing=skip_existing)

            manager['pred_trk_viz'] = Task(**{
                'name': 'pred_trk_viz',
                'requested': config['enable_pred_trk_viz'],
                'output': pred_trk_row['pred_trk_viz_stamp'],
                'requires': ['pred_trk'],
                'recompute': recompute_track,
            }, manager=manager, skip_existing=skip_existing)

            manager['eval_trk'] = Task(**{
                'name': 'eval_trk',
                'requested': with_iarpa_eval,
                'output': pred_trk_row['eval_trk_out_fpath'],
                'requires': ['pred_trk'],
                'recompute': recompute_iarpa_eval,
            }, manager=manager, skip_existing=skip_existing)

            bas_job = None
            task_info = manager['pred_trk']
            if task_info.should_compute_task():
                cfg = pred_trk_row['trk_cfg'].copy()
                if 'thresh_hysteresis' in cfg:
                    if isinstance(cfg['thresh_hysteresis'], str):
                        cfg['thresh_hysteresis'] = util_globals.restricted_eval(
                            cfg['thresh_hysteresis'].format(**cfg))

                if 'moving_window_size' in cfg:
                    if isinstance(cfg['moving_window_size'], str):
                        cfg['moving_window_size'] = util_globals.restricted_eval(
                            cfg['moving_window_size'].format(**cfg))
                    if cfg['moving_window_size'] is None:
                        cfg['polygon_fn'] = 'heatmaps_to_polys'
                    else:
                        cfg['polygon_fn'] = 'heatmaps_to_polys_moving_window'

                track_kwargs_str = shlex.quote(json.dumps(cfg))
                bas_args = f'--default_track_fn saliency_heatmaps --track_kwargs {track_kwargs_str}'
                pred_trk_row['bas_args'] = bas_args
                # Because BAS is the first step we want ensure we clear annotations so
                # everything that comes out is a track from BAS.
                command = ub.codeblock(
                    r'''
                    python -m watch.cli.kwcoco_to_geojson \
                        "{pred_pxl_fpath}" \
                        {bas_args} \
                        --clear_annots \
                        --out_dir "{pred_trk_dpath}" \
                        --out_fpath "{pred_trk_fpath}" \
                        --out_kwcoco "{pred_trk_kwcoco_fpath}"
                    ''').format(**pred_trk_row)
                command = task_info.prefix_command(command)
                name = task_info['name'] + pred_trk_row['name_suffix']
                bas_job = queue.submit(command=command, depends=pred_job,
                                         name=name, cpus=2, **common_submitkw)
                task_info['job'] = bas_job

            task_info = manager['pred_trk_viz']
            if task_info.should_compute_task():
                condensed = pred_trk_row['condensed']
                pred_trk_row['extra_header'] = f"\\n{condensed['pred_cfg']}-{condensed['trk_cfg']}"
                command = ub.codeblock(
                    r'''
                    smartwatch visualize \
                        "{pred_trk_kwcoco_fpath}" \
                        --channels="red|green|blue,salient" \
                        --stack=only \
                        --workers=avail/2 \
                        --workers=avail/2 \
                        --extra_header="{extra_header}" \
                        --animate=True && touch {pred_trk_viz_stamp}
                    ''').format(**pred_trk_row)
                print(command)
                # FIXME: the process itself should likely take care of writing
                # a stamp that indicates it is done. Or we can generalize this
                # as some wrapper applied to every watch command, but that
                # might require knowing about all configs a-priori.
                command = task_info.prefix_command(command)
                name = task_info['name'] + pred_trk_row['name_suffix']
                bas_viz_job = queue.submit(command=command, depends=bas_job,
                                           name=name, cpus=2,
                                           **common_submitkw)
                task_info['job'] = bas_viz_job

            # TODO: need a way of knowing if a package is BAS or SC.
            # Might need info on GSD as well.
            task_info = manager['eval_trk']
            if task_info.should_compute_task():
                eval_trk_row = pred_trk_row.copy()  # 1-to-1 for now
                command = ub.codeblock(
                    r'''
                    python -m watch.cli.run_metrics_framework \
                        --merge=True \
                        --true_site_dpath "{true_site_dpath}" \
                        --true_region_dpath "{true_region_dpath}" \
                        --tmp_dir "{eval_trk_tmp_dpath}" \
                        --out_dir "{eval_trk_score_dpath}" \
                        --name "{name_suffix}" \
                        --merge_fpath "{eval_trk_fpath}" \
                        --inputs_are_paths=True \
                        --pred_sites={pred_trk_fpath}
                    ''').format(**eval_trk_row)

                command = task_info.prefix_command(command)
                name = task_info['name'] + eval_trk_row['name_suffix']
                bas_eval_job = queue.submit(
                    command=command, depends=bas_job, name=name, cpus=2,
                    **common_submitkw)
                task_info['job'] = bas_eval_job

        for pred_act_row in candidate_act_rows:

            manager['pred_act'] = Task(**{
                'name': 'pred_act',
                'requested': config['enable_pred_act'],
                'output': pred_act_row['pred_act_fpath'],
                'requires': ['pred_pxl'],
                'recompute': 0,
            }, manager=manager, skip_existing=skip_existing)
            manager['eval_act'] = Task(**{
                'name': 'eval_act',
                'requested': config['enable_eval_act'],
                'output': pred_act_row['eval_act_fpath'],
                'requires': ['pred_act'],
                'recompute': 0,
            }, manager=manager, skip_existing=skip_existing)

            sc_job = None
            task_info = manager['pred_act']
            if task_info.should_compute_task():
                actclf_cfg = {
                    'boundaries_as': 'polys',
                }
                actclf_cfg.update(pred_act_row['act_cfg'])
                kwargs_str = shlex.quote(json.dumps(actclf_cfg))
                pred_act_row['sc_args'] = f'--default_track_fn class_heatmaps --track_kwargs {kwargs_str}'

                # TODO: make a variant that comes from truth, and a variant
                # that comes from a selected BAS model.
                command = ub.codeblock(
                    r'''
                    python -m watch.cli.kwcoco_to_geojson \
                        "{pred_pxl_fpath}" \
                        --site_summary '{site_summary_glob}' \
                        {sc_args} \
                        --out_dir "{pred_act_dpath}" \
                        --out_fpath "{pred_act_fpath}" \
                        --out_kwcoco_fpath "{pred_act_kwcoco_fpath}"
                    ''').format(**pred_act_row)

                command = task_info.prefix_command(command)
                name = task_info['name'] + pred_act_row['name_suffix']
                sc_job = queue.submit(
                    command=command, depends=pred_job, name=name, cpus=2,
                    partition=config['partition'],
                    mem=config['mem'],
                )
                task_info['job'] = sc_job

            task_info = manager['eval_act']
            if task_info.should_compute_task():
                eval_act_row = pred_act_row.copy()  # 1-to-1 for now
                command = ub.codeblock(
                    r'''
                    python -m watch.cli.run_metrics_framework \
                        --merge=True \
                        --true_site_dpath "{true_site_dpath}" \
                        --true_region_dpath "{true_region_dpath}" \
                        --tmp_dir "{eval_act_tmp_dpath}" \
                        --out_dir "{eval_act_score_dpath}" \
                        --name "{name_suffix}" \
                        --merge_fpath "{eval_act_fpath}" \
                        --inputs_are_paths=True \
                        --pred_sites={pred_act_fpath}
                    ''').format(**eval_act_row)

                command = task_info.prefix_command(command)
                name = task_info['name'] + eval_act_row['name_suffix']
                sc_eval_job = queue.submit(
                    command=command,
                    depends=sc_job,
                    name=name,
                    cpus=2,
                    partition=config['partition'],
                    mem=config['mem'],
                )
                task_info['job'] = sc_eval_job

    print('queue = {!r}'.format(queue))
    # print(f'{len(queue)=}')
    with_status = 0
    with_rich = 0
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


def handle_yaml_grid(default, auto, arg):
    """
    Example:
        >>> default = {}
        >>> auto = {}
        >>> arg = ub.codeblock(
        >>>     '''
        >>>     matrix:
        >>>         foo: ['bar', 'baz']
        >>>     include:
        >>>         - {'foo': 'buz', 'bug': 'boop'}
        >>>     ''')
        >>> handle_yaml_grid(default, auto, arg)

        >>> default = {'baz': [1, 2, 3]}
        >>> arg = '''
        >>>     include:
        >>>     - {
        >>>       "thresh": 0.1,
        >>>       "morph_kernel": 3,
        >>>       "norm_ord": 1,
        >>>       "agg_fn": "probs",
        >>>       "thresh_hysteresis": "None",
        >>>       "moving_window_size": "None",
        >>>       "polygon_fn": "heatmaps_to_polys"
        >>>     }
        >>>     '''
        >>> handle_yaml_grid(default, auto, arg)
    """
    stdform_keys = {'matrix', 'include'}
    import ruamel.yaml
    print('arg = {}'.format(ub.repr2(arg, nl=1)))
    if arg:
        if arg is True:
            arg = 'auto'
        if isinstance(arg, str):
            if arg == 'auto':
                arg = auto
            if isinstance(arg, str):
                arg = ruamel.yaml.safe_load(arg)
    else:
        arg = {'matrix': default}
    if isinstance(arg, dict):
        arg = ub.udict(arg)
        if len(arg - stdform_keys) == 0 and (arg & stdform_keys):
            # Standard form
            ...
        else:
            # Transform matrix to standard form
            arg = {'matrix': arg}
    elif isinstance(arg, list):
        # Transform list form to standard form
        arg = {'include': arg}
    else:
        raise TypeError(type(arg))
    assert set(arg.keys()).issubset(stdform_keys)
    print('arg = {}'.format(ub.repr2(arg, nl=1)))
    basis = arg.get('matrix', {})
    if basis:
        grid = list(ub.named_product(basis))
    else:
        grid = []
    grid.extend(arg.get('include', []))
    return grid


def handle_param_grid(arg):
    """
    Our own method for specifying many combinations. Uses the github actions
    method under the hood with our own

        >>> from watch.mlops.schedule_evaluation import *  # NOQA
        >>> arg = ub.codeblock(
            '''
            - matrix:
                trk.pxl.model: [trk_a, trk_b]
                trk.pxl.data.tta_time: [0, 4]
                trk.pxl.data.set_cover_algo: [None, approx]
                trk.pxl.data.test_dataset: [D4_S2_L8]

                act.pxl.model: [act_a, act_b]
                act.pxl.data.test_dataset: [D4_WV_PD, D4_WV]
                act.pxl.data.input_space_scale: [1GSD, 4GSD]

                trk.poly.thresh: [0.17]
                act.poly.thresh: [0.13]

                exclude:
                  #
                  # The BAS A should not run with tta
                  - trk.pxl.model: trk_a
                    trk.pxl.data.tta_time: 4
                  # The BAS B should not run without tta
                  - trk.pxl.model: trk_b
                    trk.pxl.data.tta_time: 0
                  #
                  # The SC B should not run on the PD dataset when GSD is 1
                  - act.pxl.model: act_b
                    act.pxl.data.test_dataset: D4_WV_PD
                    act.pxl.data.input_space_scale: 1GSD
                  # The SC A should not run on the WV dataset when GSD is 4
                  - act.pxl.model: act_a
                    act.pxl.data.test_dataset: D4_WV
                    act.pxl.data.input_space_scale: 4GSD
                  #
                  # The The BAS A and SC B model should not run together
                  - trk.pxl.model: trk_a
                    act.pxl.model: act_b
                  # Other misc exclusions to make the output cleaner
                  - trk.pxl.model: trk_b
                    act.pxl.data.input_space_scale: 4GSD
                  - trk.pxl.data.set_cover_algo: None
                    act.pxl.data.input_space_scale: 1GSD

                include:
                  # only try the 10GSD scale for trk model A
                  - trk.pxl.model: trk_a
                    trk.pxl.data.input_space_scale: 10GSD
            ''')
        >>> grid_items = handle_param_grid(arg)
        >>> print('grid_items = {}'.format(ub.repr2(grid_items, nl=1, sort=0)))
        >>> print(ub.repr2([dotdict_to_nested(p) for p in grid_items], nl=-3, sort=0))
        >>> print(len(grid_items))
    """
    import ruamel.yaml
    if isinstance(arg, str):
        data = ruamel.yaml.safe_load(arg)
    else:
        data = arg.copy()
    grid_items = []
    if isinstance(data, list):
        for item in data:
            grid_items += github_action_matrix(item)
    elif isinstance(data, dict):
        grid_items += github_action_matrix(data)
    return grid_items


def dotdict_to_nested(d):
    auto = ub.AutoDict()
    walker = ub.IndexableWalker(auto)
    for k, v in d.items():
        path = k.split('.')
        walker[path] = v
    return auto.to_dict()


def github_action_matrix(arg):
    """
    Try to implement the github method. Not sure if I like it.

    References:
        https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs#expanding-or-adding-matrix-configurations

    Example:
        >>> from watch.mlops.schedule_evaluation import *  # NOQA
        >>> arg = ub.codeblock(
                 '''
                   matrix:
                     fruit: [apple, pear]
                     animal: [cat, dog]
                     include:
                       - color: green
                       - color: pink
                         animal: cat
                       - fruit: apple
                         shape: circle
                       - fruit: banana
                       - fruit: banana
                         animal: cat
                 ''')
        >>> grid_items = github_action_matrix(arg)
        >>> print('grid_items = {}'.format(ub.repr2(grid_items, nl=1)))
        grid_items = [
            {'animal': 'cat', 'color': 'pink', 'fruit': 'apple', 'shape': 'circle'},
            {'animal': 'dog', 'color': 'green', 'fruit': 'apple', 'shape': 'circle'},
            {'animal': 'cat', 'color': 'pink', 'fruit': 'pear'},
            {'animal': 'dog', 'color': 'green', 'fruit': 'pear'},
            {'fruit': 'banana'},
            {'animal': 'cat', 'fruit': 'banana'},
        ]


    Example:
        >>> from watch.mlops.schedule_evaluation import *  # NOQA
        >>> arg = ub.codeblock(
                '''
                  matrix:
                    os: [macos-latest, windows-latest]
                    version: [12, 14, 16]
                    environment: [staging, production]
                    exclude:
                      - os: macos-latest
                        version: 12
                        environment: production
                      - os: windows-latest
                        version: 16
            ''')
        >>> grid_items = github_action_matrix(arg)
        >>> print('grid_items = {}'.format(ub.repr2(grid_items, nl=1)))
        grid_items = [
            {'environment': 'staging', 'os': 'macos-latest', 'version': 12},
            {'environment': 'staging', 'os': 'macos-latest', 'version': 14},
            {'environment': 'production', 'os': 'macos-latest', 'version': 14},
            {'environment': 'staging', 'os': 'macos-latest', 'version': 16},
            {'environment': 'production', 'os': 'macos-latest', 'version': 16},
            {'environment': 'staging', 'os': 'windows-latest', 'version': 12},
            {'environment': 'production', 'os': 'windows-latest', 'version': 12},
            {'environment': 'staging', 'os': 'windows-latest', 'version': 14},
            {'environment': 'production', 'os': 'windows-latest', 'version': 14},
        ]
    """
    import ruamel.yaml
    if isinstance(arg, str):
        data = ruamel.yaml.safe_load(arg)
    else:
        data = arg.copy()

    matrix = data.pop('matrix')
    include = [ub.udict(p) for p in matrix.pop('include', [])]
    exclude = [ub.udict(p) for p in matrix.pop('exclude', [])]

    matrix_ = {k: (v if ub.iterable(v) else [v])
               for k, v in matrix.items()}
    grid_stage0 = list(map(ub.udict, ub.named_product(matrix_)))

    # Note: All include combinations are processed after exclude. This allows
    # you to use include to add back combinations that were previously
    # excluded.
    def is_excluded(grid_item):
        for exclude_item in exclude:
            common1 = exclude_item & grid_item
            if common1:
                common2 = grid_item & exclude_item
                if common1 == common2 == exclude_item:
                    return True

    grid_stage1 = [p for p in grid_stage0 if not is_excluded(p)]

    orig_keys = set(matrix.keys())
    # Extra items are never modified by future include values include values
    # will only modify non-conflicting original grid items or create one of
    # these special immutable grid items.
    appended_items = []

    # For each object in the include list
    for include_item in include:
        any_updated = False
        for grid_item in grid_stage1:
            common_orig1 = (grid_item & include_item) & orig_keys
            common_orig2 = (include_item & grid_item) & orig_keys
            if common_orig1 == common_orig2:
                # the key:value pairs in the object will be added to each of
                # the [original] matrix combinations if none of the key:value
                # pairs overwrite any of the original matrix values
                any_updated = True
                # Note that the original matrix values will not be overwritten
                # but added matrix values can be overwritten
                grid_item.update(include_item)
        if not any_updated:
            # If the object cannot be added to any of the matrix combinations, a
            # new matrix combination will be created instead.
            appended_items.append(include_item)
    grid_items = grid_stage1 + appended_items

    return grid_items


if __name__ == '__main__':
    schedule_evaluation(cmdline=True)

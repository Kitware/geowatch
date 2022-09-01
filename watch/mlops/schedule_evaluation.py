"""
Helper for scheduling a set of prediction + evaluation jobs

python -m watch.tasks.fusion.schedule_evaluation

DVC_DPATH=$(smartwatch_dvc)
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
EXPT_PATTERN="*"
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
python -m watch.tasks.fusion.schedule_evaluation \
        --devices="0,1" \
        --model_globstr="$DVC_DPATH/models/fusion/eval3_candidates/packages/${EXPT_PATTERN}/*.pt" \
        --test_dataset="$VALI_FPATH" \
        --run=0 --skip_existing=True
"""
import ubelt as ub
import scriptconfig as scfg


class ScheduleEvaluationConfig(scfg.Config):
    """
    Builds commands and optionally schedules them.
    """
    default = {
        'model_globstr': scfg.Value(None, help='one or more glob patterns that match the models to predict/evaluate on'),
        'test_dataset': scfg.Value(None, help='path to the test dataset to predict/evaluate on'),
        'devices': scfg.Value('auto', help='if using tmux or serial, indicate which gpus are available for use as a comma separated list: e.g. 0,1'),
        'run': scfg.Value(False, help='if False, only prints the commands, otherwise executes them'),
        'virtualenv_cmd': scfg.Value(None, help='command to activate a virtualenv if needed. (might have issues with slurm backend)'),
        'skip_existing': scfg.Value(False, help='if True dont submit commands where the expected products already exist'),
        'backend': scfg.Value('tmux', help='can be tmux, slurm, or maybe serial in the future'),

        'pred_workers': scfg.Value(4, help='number of prediction workers in each process'),

        'enable_eval': scfg.Value(True, help='if False, then evaluation is not run'),
        'enable_pred': scfg.Value(True, help='if False, then prediction is not run'),

        'draw_heatmaps': scfg.Value(1, help='if true draw heatmaps on eval'),
        'draw_curves': scfg.Value(1, help='if true draw curves on eval'),

        'partition': scfg.Value(None, help='specify slurm partition (slurm backend only)'),
        'mem': scfg.Value(None, help='specify slurm memory per task (slurm backend only)'),

        'tta_fliprot': scfg.Value(None, help='grid of flip test-time-augmentation to test'),
        'tta_time': scfg.Value(None, help='grid of temporal test-time-augmentation to test'),
        'chip_overlap': scfg.Value(0.3, help='grid of chip overlaps test'),
        'set_cover_algo': scfg.Value(['approx'], help='grid of set_cover_algo to test'),

        'sidecar2': scfg.Value(True, help='if True uses parallel sidecar pattern, otherwise nested'),

        'shuffle_jobs': scfg.Value(True, help='if True, shuffles the jobs so they are submitted in a random order'),

        'enable_track': scfg.Value(False, help='if True, enable tracking'),
        'enable_iarpa_eval': scfg.Value(False, help='if True, enable iapra BAS evalaution'),

        'enable_actclf': scfg.Value(False, help='if True, enable actclf'),
        'enable_actclf_eval': scfg.Value(False, help='if True, enable iapra SC evalaution'),

        'annotations_dpath': scfg.Value(None, help='path to IARPA annotations dpath for IARPA eval'),

        'bas_thresh': scfg.Value([0.01], help='grid of track thresholds'),

        'hack_bas_grid': scfg.Value(False, help='if True use hard coded BAS grid'),
        'hack_sc_grid': scfg.Value(False, help='if True use hard coded SC grid'),
        'dvc_expt_dpath': None,
        'dvc_data_dpath': None,
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

        if not all_deps_will_exist:
            # If dependencies wont exist, then we cant run
            enabled = False
        else:
            # If we can run, then do it this task is requested
            if task_info.skip_existing and task_info['output'].exists():
                enabled = task_info['recompute']
            else:
                enabled = bool(task_info['requested'])
        # Only enable the task if we requested it and its dependencies will
        # exist.
        task_info['enabled'] = enabled
        # Mark if we do exist, or we will exist
        will_exist = enabled or task_info['output'].exists()
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


def schedule_evaluation(cmdline=False, **kwargs):
    """
    First ensure that models have been copied to the DVC repo in the
    appropriate path. (as noted by model_dpath)
    """
    import watch
    import shlex
    import json
    from watch.utils.lightning_ext import util_globals
    import glob
    import kwarray
    import cmd_queue

    config = ScheduleEvaluationConfig(cmdline=cmdline, data=kwargs)
    print('ScheduleEvaluationConfig config = {}'.format(ub.repr2(dict(config), nl=1)))

    model_globstr = config['model_globstr']
    test_dataset_fpath = config['test_dataset']
    draw_curves = config['draw_curves']
    draw_heatmaps = config['draw_heatmaps']
    dvc_expt_dpath = config['dvc_expt_dpath']
    dvc_data_dpath = config['dvc_data_dpath']

    if model_globstr is None and test_dataset_fpath is None:
        raise ValueError('model_globstr and test_dataset are required')

    # HACK FOR DVC PTH FIXME:
    if str(model_globstr).endswith('.txt'):
        from watch.utils.simple_dvc import SimpleDVC
        print('model_globstr = {!r}'.format(model_globstr))
        if dvc_expt_dpath is None:
            dvc_expt_dpath = SimpleDVC.find_root(ub.Path(model_globstr))

    if dvc_expt_dpath is None:
        dvc_expt_dpath = watch.find_smart_dvc_dpath(tags='phase2_expt')
    if dvc_data_dpath is None:
        dvc_data_dpath = watch.find_smart_dvc_dpath(tags='phase2_data')
    dvc_data_dpath = ub.Path(dvc_data_dpath)
    dvc_expt_dpath = ub.Path(dvc_expt_dpath)

    print(f'dvc_expt_dpath={dvc_expt_dpath}')
    print(f'dvc_data_dpath={dvc_data_dpath}')

    with_saliency = 'auto'
    with_class = 'auto'

    skip_existing = config['skip_existing']
    with_pred = config['enable_pred']  # TODO: allow caching
    with_eval = config['enable_eval']
    with_track = config['enable_track']
    with_iarpa_eval = config['enable_iarpa_eval']
    recompute = False

    def check_recompute(flag, depends_flags=[]):
        return recompute or flag == 'redo' or any(f == 'redo' for f in depends_flags)
    recompute_pred = check_recompute(with_pred)
    recompute_eval = check_recompute(with_pred, [with_pred])
    recompute_track = check_recompute(with_track, [with_pred])
    recompute_iarpa_eval = check_recompute(with_iarpa_eval, [with_pred, recompute_track])
    print('with_pred = {!r}'.format(with_pred))
    print('with_pred = {!r}'.format(with_pred))
    print('with_track = {!r}'.format(with_track))
    print('with_iarpa_eval = {!r}'.format(with_iarpa_eval))

    print('recompute_pred = {!r}'.format(recompute_pred))
    print('recompute_eval = {!r}'.format(recompute_eval))
    print('recompute_track = {!r}'.format(recompute_track))
    print('recompute_iarpa_eval = {!r}'.format(recompute_iarpa_eval))

    workers_per_queue = config['pred_workers']

    test_dataset_fpath = ub.Path(test_dataset_fpath)
    if not test_dataset_fpath.exists():
        print('warning test dataset does not exist')

    annotations_dpath = config['annotations_dpath']
    if annotations_dpath is None:
        annotations_dpath = dvc_data_dpath / 'annotations'
    annotations_dpath = ub.Path(annotations_dpath)
    region_model_dpath = annotations_dpath / 'region_models'

    def expand_model_list_file(model_lists_fpath, dvc_expt_dpath=None):
        """
        Given a file containing paths to models, expand it into individual
        paths.
        """
        expanded_fpaths = []
        lines = [line for line in ub.Path(model_globstr).read_text().split('\n') if line]
        missing = []
        for line in lines:
            if dvc_expt_dpath is not None:
                package_fpath = ub.Path(dvc_expt_dpath / line)
            else:
                package_fpath = ub.Path(line)
            if package_fpath.is_file():
                expanded_fpaths.append(package_fpath)
            else:
                missing.append(line)
        if missing:
            print('WARNING: missing = {}'.format(ub.repr2(missing, nl=1)))
            print(f'WARNING: specified a models-of-interest.txt and {len(missing)} / {len(lines)} models were missing')
        return expanded_fpaths

    print('model_globstr = {!r}'.format(model_globstr))
    package_fpaths = []
    for package_fpath in glob.glob(model_globstr, recursive=True):
        package_fpath = ub.Path(package_fpath)
        if package_fpath.name.endswith('.txt'):
            # HACK FOR PATH OF MODELS
            model_lists_fpath = package_fpath
            expanded_fpaths = expand_model_list_file(model_lists_fpath, dvc_expt_dpath=dvc_expt_dpath)
            package_fpaths.extend(expanded_fpaths)
        else:
            package_fpaths.append(package_fpath)

    if len(package_fpaths) == 0:
        if '*' not in str(model_globstr):
            package_fpaths = [ub.Path(model_globstr)]

    from watch.mlops.expt_manager import ExperimentState
    # start using the experiment state logic as the path and metadata
    # organization logic
    state = ExperimentState('*', '*')
    candidate_pkg_rows = []
    for package_fpath in package_fpaths:
        condensed = state._parse_pattern_attrs('pkg', package_fpath)
        package_info = {}
        package_info['package_fpath'] = package_fpath
        package_info['condensed'] = condensed
        candidate_pkg_rows.append(package_info)
    print(f'{len(candidate_pkg_rows)=}')

    queue_dpath = dvc_expt_dpath / '_cmd_queue_schedule'
    queue_dpath.mkdir(exist_ok=True)

    devices = config['devices']
    print('devices = {!r}'.format(devices))
    if devices == 'auto':
        GPUS = _auto_gpus()
    else:
        GPUS = None if devices is None else ensure_iterable(devices)
    print('GPUS = {!r}'.format(GPUS))

    environ = {}
    queue = cmd_queue.Queue.create(config['backend'], name='schedule-eval',
                                   size=len(GPUS), environ=environ,
                                   dpath=queue_dpath, gres=GPUS)

    virtualenv_cmd = config['virtualenv_cmd']
    if virtualenv_cmd:
        queue.add_header_command(virtualenv_cmd)

    # Define the parameter grids to loop over

    pred_cfg_basis = {}
    pred_cfg_basis['tta_time'] = ensure_iterable(config['tta_time'])
    pred_cfg_basis['tta_fliprot'] = ensure_iterable(config['tta_fliprot'])
    pred_cfg_basis['chip_overlap'] = ensure_iterable(config['chip_overlap'])
    pred_cfg_basis['set_cover_algo'] = ensure_iterable(config['set_cover_algo'])

    trk_defaults = {
        'thresh': [0.1],
        'morph_kernel': [3],
        'norm_ord': [1],
        'agg_fn': ['probs'],
        'thresh_hysteresis': [None],
        'moving_window_size': [None],
    }
    trk_param_basis = trk_defaults.copy()
    trk_param_basis.update({
        'thresh': ensure_iterable(config['bas_thresh']),
        # 'thresh': [0.1, 0.2, 0.3],
    })
    if config['hack_bas_grid']:
        grid = {
            'thresh': [0.01, 0.05, 0.1, 0.15, 0.2],
            'morph_kernel': [3],
            'norm_ord': [1],
            'agg_fn': ['probs', 'mean_normalized'],
            'thresh_hysteresis': [None, '2*{thresh}'],
            'moving_window_size': [None, 150],
        }
        trk_param_basis.update(grid)

    act_param_basis = {
        # TODO viterbi or not
        # Not sure what SC thresh is
        'thresh': ensure_iterable(config['bas_thresh']),
        # 'thresh': [0.0],
        'use_viterbi': [0],
    }
    if config['hack_sc_grid']:
        # TODO: remove
        grid = {
            'thresh': [0, 0.01, 0.1],
            # 'use_viterbi': [0],
            'use_viterbi': [0, 'v1,v6'],
        }
        act_param_basis.update(grid)

    # Build the info we need to submit every prediction job of interest
    candidate_pred_rows = []
    test_dset = state._condense_test_dset(test_dataset_fpath)
    for pkg_row in candidate_pkg_rows:
        for pred_cfg in ub.named_product(pred_cfg_basis):
            pred_pxl_row = pkg_row.copy()
            condensed  = pred_pxl_row['condensed'].copy()

            condensed['test_dset'] = test_dset
            condensed['pred_cfg'] = state._condense_pred_cfg(pred_cfg)

            pred_pxl_row['condensed'] = condensed
            pred_pxl_row['pred_cfg'] = pred_cfg
            pred_pxl_row['test_dataset_fpath'] = test_dataset_fpath
            # TODO: make using these templates easier
            pred_pxl_row['pred_dataset_fpath'] = ub.Path(state.templates['pred_pxl'].format(**condensed))
            pred_pxl_row['eval_pxl_fpath'] = ub.Path(state.templates['eval_pxl'].format(**condensed))
            pred_pxl_row['eval_pxl_dpath'] = pred_pxl_row['eval_pxl_fpath'].parent.parent

            # TODO: Technically the there should be an evaluation config list
            # that is looped over for every prediction, but for now they are
            # 1-to-1. A general ML-ops framwork should provide this.

            # TODO: these are really part of the pred_cfg, even though they are
            # semi-non-impactful, handle them gracefully
            pred_pxl_row['with_class'] = with_class
            pred_pxl_row['with_saliency'] = with_saliency
            # These are part of the pxl eval config
            pred_pxl_row['draw_curves'] = draw_curves
            pred_pxl_row['draw_heatmaps'] = draw_heatmaps
            candidate_pred_rows.append(pred_pxl_row)

    if with_eval == 'redo':
        # Need to dvc unprotect
        # TODO: this can be a job in the queue
        needs_unprotect = []
        for pred_pxl_row in candidate_pred_rows:
            eval_metrics_fpath = ub.Path(pred_pxl_row['eval_pxl_fpath'])
            eval_metrics_dvc_fpath = eval_metrics_fpath.augment(tail='.dvc')
            if eval_metrics_dvc_fpath.exists():
                needs_unprotect.append(eval_metrics_fpath)
        if needs_unprotect:
            # TODO: use the dvc experiment manager for this.
            # This should not be our concern
            from watch.utils.simple_dvc import SimpleDVC
            simple_dvc = SimpleDVC(dvc_expt_dpath)
            simple_dvc.unprotect(needs_unprotect)

    if config['shuffle_jobs']:
        candidate_pred_rows = kwarray.shuffle(candidate_pred_rows)

    common_submitkw = dict(
        partition=config['partition'],
        mem=config['mem']
    )

    for pred_pxl_row in candidate_pred_rows:
        package_fpath = pred_pxl_row['package_fpath']
        pred_cfg = pred_pxl_row['pred_cfg']
        condensed = pred_pxl_row['condensed']
        pred_dataset_fpath = ub.Path(pred_pxl_row['pred_dataset_fpath'])
        eval_pxl_fpath = ub.Path(pred_pxl_row['eval_pxl_fpath'])
        pred_pxl_row['name_suffix'] = pred_pxl_row['name_suffix'] = '-'.join([
            condensed['model'],
            condensed['pred_cfg'],
        ])

        # First compute children track, activity rows (todo: refactor to do
        # ealier)
        trk_rows = []
        for trk_cfg in ub.named_product(trk_param_basis):
            pred_trk_row = pred_pxl_row.copy()
            pred_trk_row['condensed'] = condensed = pred_trk_row['condensed'].copy()
            condensed['trk_cfg'] = state._condense_trk_cfg(trk_cfg)
            pred_trk_row['name_suffix'] = '-'.join([
                condensed['model'],
                condensed['pred_cfg'],
                condensed['trk_cfg'],
            ])

            pred_trk_row['trk_pred_fpath'] = ub.Path(state.templates['pred_trk'].format(**condensed))
            pred_trk_row['bas_eval_out_fpath'] = ub.Path(state.templates['eval_trk'].format(**condensed))
            pred_trk_row['trk_eval_fpath'] = ub.Path(state.templates['eval_trk'].format(**condensed))
            pred_trk_row['trk_score_dpath'] = pred_trk_row['trk_eval_fpath'].parent.parent
            pred_trk_row['trk_eval_dpath'] = pred_trk_row['trk_score_dpath'].parent
            pred_trk_row['pred_tracks_dpath'] = pred_trk_row['trk_pred_fpath'].parent / 'tracked_sites'

            pred_trk_row['trk_tmp_dpath'] = pred_trk_row['trk_eval_dpath'] / '_tmp'
            pred_trk_row['true_site_dpath'] = annotations_dpath / 'site_models'
            pred_trk_row['true_region_dpath'] = annotations_dpath / 'region_models'
            pred_trk_row['trk_cfg'] = trk_cfg
            trk_rows.append(pred_trk_row)

        act_rows = []
        for act_cfg in ub.named_product(act_param_basis):
            pred_act_row = pred_pxl_row.copy()
            pred_act_row['condensed'] = condensed = pred_act_row['condensed'].copy()
            condensed['act_cfg'] = state._condense_act_cfg(act_cfg)
            pred_act_row['act_pred_fpath'] = ub.Path(state.templates['pred_act'].format(**condensed))
            pred_act_row['act_pred_dpath'] = pred_act_row['act_pred_fpath'].parent / 'classified_sites'

            pred_act_row['act_eval_fpath'] = ub.Path(state.templates['eval_act'].format(**condensed))
            pred_act_row['act_score_dpath'] = pred_act_row['act_eval_fpath'].parent.parent
            pred_act_row['act_eval_dpath'] = pred_act_row['act_score_dpath'].parent
            pred_act_row['site_summary_glob'] = (region_model_dpath / '*.geojson')

            pred_act_row['true_site_dpath'] = annotations_dpath / 'site_models'
            pred_act_row['true_region_dpath'] = annotations_dpath / 'region_models'
            pred_act_row['act_tmp_dpath'] = pred_act_row['act_eval_dpath'] / '_tmp'
            pred_act_row['act_cfg'] = act_cfg

            pred_act_row['name_suffix'] = '-'.join([
                condensed['model'],
                condensed['pred_cfg'],
                condensed['act_cfg'],
            ])
            act_rows.append(pred_act_row)

        # Really should make this a class
        manager = {}
        manager['pred'] = Task(**{
            'name': 'pred',
            'requested': with_pred,
            'output': pred_dataset_fpath,
            'requires': [],
            'recompute': recompute_pred,
        }, manager=manager, skip_existing=skip_existing)

        manager['pxl_eval'] = Task(**{
            'name': 'pxl_eval',
            'requested': with_eval,
            'output': eval_pxl_fpath,
            'requires': ['pred'],
            'recompute': recompute_eval,
        }, manager=manager, skip_existing=skip_existing)

        pred_job = None
        task_info = manager['pred']
        if task_info.should_compute_task():
            predictkw = {
                'workers_per_queue': workers_per_queue,
                **pred_cfg,
                **pred_pxl_row,
            }
            command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.predict \
                    --write_probs=True \
                    --write_preds=False \
                    --with_class={with_class} \
                    --with_saliency={with_saliency} \
                    --with_change=False \
                    --package_fpath={package_fpath} \
                    --pred_dataset={pred_dataset_fpath} \
                    --test_dataset={test_dataset_fpath} \
                    --num_workers={workers_per_queue} \
                    --chip_overlap={chip_overlap} \
                    --tta_time={tta_time} \
                    --tta_fliprot={tta_fliprot} \
                    --set_cover_algo={set_cover_algo} \
                    --devices=0, \
                    --accelerator=gpu \
                    --batch_size=1
                ''').format(**predictkw)
            command = task_info.prefix_command(command)
            name = task_info['name'] + pred_pxl_row['name_suffix']
            pred_cpus = workers_per_queue
            pred_job = queue.submit(command, gpus=1, name=name,
                                    cpus=pred_cpus, **common_submitkw)

        task_info = manager['pxl_eval']
        if task_info.should_compute_task():

            pxl_eval_row = pred_pxl_row  # hack, for now these are 1-to-1

            command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.evaluate \
                    --true_dataset={test_dataset_fpath} \
                    --pred_dataset={pred_dataset_fpath} \
                      --eval_dpath={eval_pxl_dpath} \
                      --score_space=video \
                      --draw_curves={draw_curves} \
                      --draw_heatmaps={draw_heatmaps} \
                      --workers=2
                ''').format(**pxl_eval_row)
            command = task_info.prefix_command(command)
            name = task_info['name'] + pred_pxl_row['name_suffix']
            task_info['job'] = queue.submit(
                command, depends=pred_job, name=name, cpus=2,
                **common_submitkw)

        for pred_trk_row in trk_rows:
            manager['bas_track'] = Task(**{
                'name': 'bas_track',
                'requested': with_track,
                'output': pred_trk_row['trk_pred_fpath'],
                'requires': ['pred'],
                'recompute': recompute_track,
            }, manager=manager, skip_existing=skip_existing)
            manager['bas_eval'] = Task(**{
                'name': 'bas_eval',
                'requested': with_iarpa_eval,
                'output': pred_trk_row['bas_eval_out_fpath'],
                'requires': ['bas_track'],
                'recompute': recompute_iarpa_eval,
            }, manager=manager, skip_existing=skip_existing)

            bas_job = None
            task_info = manager['bas_track']
            if task_info.should_compute_task():
                cfg = pred_trk_row['trk_cfg'].copy()
                if isinstance(cfg['thresh_hysteresis'], str):
                    cfg['thresh_hysteresis'] = util_globals.restricted_eval(
                        cfg['thresh_hysteresis'].format(**cfg))

                if cfg['moving_window_size'] is None:
                    cfg['moving_window_size'] = 'heatmaps_to_polys'
                else:
                    cfg['moving_window_size'] = 'heatmaps_to_polys_moving_window'

                track_kwargs_str = shlex.quote(json.dumps(cfg))
                bas_args = f'--default_track_fn saliency_heatmaps --track_kwargs {track_kwargs_str}'
                pred_trk_row['bas_args'] = bas_args
                # Because BAS is the first step we want ensure we clear annotations so
                # everything that comes out is a track from BAS.
                command = ub.codeblock(
                    r'''
                    python -m watch.cli.kwcoco_to_geojson \
                        "{pred_dataset_fpath}" \
                        {bas_args} \
                        --clear_annots \
                        --out_dir "{pred_tracks_dpath}" \
                        --out_fpath "{trk_pred_fpath}"
                    ''').format(pred_trk_row)

                command = task_info.prefix_command(command)
                name = task_info['name'] + pred_trk_row['name_suffix']
                bas_job = queue.submit(command=command, depends=pred_job,
                                         name=name, cpus=2, **common_submitkw)
                task_info['job'] = bas_job

            # TODO: need a way of knowing if a package is BAS or SC.
            # Might need info on GSD as well.
            task_info = manager['bas_eval']
            if task_info.should_compute_task():
                eval_trk_row = pred_trk_row.copy()  # 1-to-1 for now
                command = ub.codeblock(
                    r'''
                    python -m watch.cli.run_metrics_framework \
                        --merge=True \
                        --true_site_dpath "{true_site_dpath}" \
                        --true_region_dpath "{true_region_dpath}" \
                        --tmp_dir "{trk_tmp_dpath}" \
                        --out_dir "{trk_score_dpath}" \
                        --name {shlex.quote(str(name_suffix))} \
                        --merge_fpath "{trk_eval_fpath}" \
                        --inputs_are_paths=True \
                        --pred_sites={trk_pred_fpath}
                    ''').format(eval_trk_row)

                command = task_info.prefix_command(command)
                name = task_info['name'] + eval_trk_row['name_suffix']
                bas_eval_job = queue.submit(
                    command=command, depends=bas_job, name=name, cpus=2,
                    **common_submitkw)
                task_info['job'] = bas_eval_job

        for pred_act_row in act_rows:

            manager['actclf'] = Task(**{
                'name': 'actclf',
                'requested': config['enable_actclf'],
                'output': pred_act_row['act_pred_fpath'],
                'requires': ['pred'],
                'recompute': 0,
            }, manager=manager, skip_existing=skip_existing)
            manager['sc_eval'] = Task(**{
                'name': 'sc_eval',
                'requested': config['enable_actclf_eval'],
                'output': pred_act_row['act_eval_fpath'],
                'requires': ['actclf'],
                'recompute': 0,
            }, manager=manager, skip_existing=skip_existing)

            sc_job = None
            task_info = manager['actclf']
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
                        "{pred_dataset_fpath}" \
                        --site_summary '{site_summary_glob}' \
                        {sc_args} \
                        --out_dir "{act_pred_dpath}" \
                        --out_fpath "{act_pred_fpath}"
                    ''').format(**pred_act_row)

                command = task_info.prefix_command(command)
                name = task_info['name'] + pred_act_row['name_suffix']
                sc_job = queue.submit(
                    command=command, depends=pred_job, name=name, cpus=2,
                    partition=config['partition'],
                    mem=config['mem'],
                )
                task_info['job'] = sc_job

            task_info = manager['sc_eval']
            if task_info.should_compute_task():
                eval_act_row = pred_act_row.copy()  # 1-to-1 for now
                command = ub.codeblock(
                    r'''
                    python -m watch.cli.run_metrics_framework \
                        --merge=True \
                        --true_site_dpath "{true_site_dpath}" \
                        --true_region_dpath "{true_region_dpath}" \
                        --tmp_dir "{act_tmp_dpath}" \
                        --out_dir "{act_score_dpath}" \
                        --name {shlex.quote(str(name))} \
                        --merge_fpath "{act_eval_fpath}" \
                        --inputs_are_paths=True \
                        --pred_sites={track_out_fpath}
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

    # RUN
    if config['run']:
        # ub.cmd('bash ' + str(driver_fpath), verbose=3, check=True)
        queue.run(block=True)
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


def package_metadata(package_fpath):
    # Hack for choosing one model from this "type"
    try:
        epoch_num = int(package_fpath.name.split('epoch=')[1].split('-')[0])
        expt_name = package_fpath.name.split('_epoch')[0]
    except Exception:
        # Try to read package metadata
        if package_fpath.exists():
            try:
                pkg_zip = ub.zopen(package_fpath, ext='.pt')
                namelist = pkg_zip.namelist()
            except Exception:
                print(f'ERROR {package_fpath=} failed to open')
                raise
            found = None
            for member in namelist:
                # if member.endswith('model.pkl'):
                if member.endswith('fit_config.yaml'):
                    found = member
                    break
            if not found:
                raise Exception(f'{package_fpath=} does not conform to name spec and does not seem to be a torch package with a package_header.json file')
            else:
                import yaml
                config_file = ub.zopen(package_fpath / found, mode='r', ext='.pt')
                config = yaml.safe_load(config_file)
                expt_name = config['name']
                # No way to introspect this (yet), so hack it
                epoch_num = -1
        else:
            epoch_num = -1
            expt_name = package_fpath.name

    info = {
        'name': expt_name,
        'epoch': epoch_num,
        'fpath': package_fpath,
    }
    return info


if __name__ == '__main__':
    schedule_evaluation(cmdline=True)

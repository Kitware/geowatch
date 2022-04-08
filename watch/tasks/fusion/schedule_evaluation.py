"""
TODO:
    - [ ] Rename to schedule inference or evaluation? Or split up the jobs?

Helper for scheduling a set of prediction + evaluation jobs

python -m watch.tasks.fusion.schedule_evaluation


DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
EXPT_PATTERN="*"
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_nowv_vali.kwcoco.json
python -m watch.tasks.fusion.schedule_evaluation \
        --gpus="0,1" \
        --model_globstr="$DVC_DPATH/models/fusion/eval3_candidates/packages/${EXPT_PATTERN}/*.pt" \
        --test_dataset="$VALI_FPATH" \
        --run=0 --skip_existing=True

# Note: change backend to tmux if slurm is not installed
DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPT_GROUP_CODE=eval3_candidates
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
        --gpus="0,1,2,3" \
        --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*.pt" \
        --test_dataset="$VALI_FPATH" \
        --run=1 --skip_existing=True --backend=slurm \
        --enable_pred=False \
        --enable_eval=redo \
        --draw_heatmaps=False

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
        'gpus': scfg.Value('auto', help='if using tmux or serial, indicate which gpus are available for use as a comma separated list: e.g. 0,1'),
        'run': scfg.Value(False, help='if False, only prints the commands, otherwise executes them'),
        'virtualenv_cmd': scfg.Value(None, help='command to activate a virtualenv if needed. (might have issues with slurm backend)'),
        'skip_existing': scfg.Value(False, help='if True dont submit commands where the expected products already exist'),
        'backend': scfg.Value('tmux', help='can be tmux, slurm, or maybe serial in the future'),

        'enable_eval': scfg.Value(True, help='if False, then evaluation is not run'),
        'enable_pred': scfg.Value(True, help='if False, then prediction is not run'),

        'draw_heatmaps': scfg.Value(True, help='if true draw heatmaps on eval'),
        'draw_curves': scfg.Value(True, help='if true draw curves on eval'),

        'partition': scfg.Value(None, help='specify slurm partition (slurm backend only)'),
        'mem': scfg.Value(None, help='specify slurm memory per task (slurm backend only)'),

        'tta_fliprot': scfg.Value(None, help='grid of flip test-time-augmentation to test'),
        'tta_time': scfg.Value(None, help='grid of temporal test-time-augmentation to test'),
        'chip_overlap': scfg.Value(0.3, help='grid of chip overlaps test'),

        'workdir': scfg.Value(None, help='if specified, dumps predictions/results here, otherwise uses our DVC sidecar pattern'),

        'sidecar2': scfg.Value(True, help='if True uses parallel sidecar pattern, otherwise nested'),

        'shuffle_jobs': scfg.Value(True, help='if True, shuffles the jobs so they are submitted in a random order'),

        'enable_iarpa_eval': scfg.Value(False, help='if True, enable iapra BAS evalaution'),
        'enable_track': scfg.Value(False, help='if True, enable tracking'),
        'annotations_dpath': scfg.Value(None, help='path to IARPA annotations dpath for IARPA eval'),
    }


def schedule_evaluation(cmdline=False, **kwargs):
    """
    First ensure that models have been copied to the DVC repo in the
    appropriate path. (as noted by model_dpath)

    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json

    kwcoco subset $DVC_DPATH/Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json
    smartwatch stats --src $DVC_DPATH/Drop1-Aligned-L1/combo_train_nowv.kwcoco.json

    # Hack to test on the train set for a sanity check
    kwcoco subset --src $DVC_DPATH/Drop1-Aligned-L1/combo_train_nowv.kwcoco.json \
            --dst $DVC_DPATH/Drop1-Aligned-L1/combo_train_US_R001_small_nowv.kwcoco.json \
            --select_videos '.name | startswith("US_R001")' \
            --select_images '.id % 4 == 0'
    smartwatch stats $DVC_DPATH/Drop1-Aligned-L1/combo_train_US_R001_small_nowv.kwcoco.json

    MODEL_GLOB=$DVC_DPATH/'models/fusion/SC-20201117/*/*.pt'
    echo "$MODEL_GLOB"

    cd $DVC_DPATH
    dvc pull -r aws --recursive models/fusion/SC-20201117

        --model_globstr="$MODEL_GLOB"
        --test_dataset="$KWCOCO_TEST_FPATH"

    CommandLine:

        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01/combo_DILM.kwcoco_vali.json

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01/combo_DILM_nowv_vali.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/SC-20201117/*/*.pt" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=0  --skip_existing=True

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01/combo_DILM_nowv_vali.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/SC-20201117/BAS_*v53*/*.pt" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=True

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-TA1-2022-01/vali_data_nowv.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/SC-20201117/BAS_*/*.pt" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=0

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01/combo_DILM_nowv_vali.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="special:HISTORY" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=1

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01/vali_data_nowv.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/SC-20201117/SC_*/*.pt" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=0 --skip_existing=1

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv_vali.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/SC-20201117/*xfer*/*.pt" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=0 --skip_existing=True

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv_vali.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/SC-20201117/SC_TA1_*/*.pt" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=1 --skip_existing=True

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_TEST_FPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv_vali.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0," \
            --model_globstr="$DVC_DPATH/models/fusion/SC-20201117/BAS_TA1_v08*/*.pt" \
            --test_dataset="$KWCOCO_TEST_FPATH" \
            --run=0 --skip_existing=True

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        KWCOCO_ALL_FPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L_nowv.kwcoco.json
        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="special:HARDCODED" \
            --test_dataset="$KWCOCO_ALL_FPATH" \
            --run=0 --skip_existing=True

    TODO:
        - [ ] Specify the model_dpath as an arg
        - [ ] Specify target dataset as an argument
        - [ ] Skip models that were already evaluated
    """
    import watch
    from watch.tasks.fusion import organize
    # import json

    config = ScheduleEvaluationConfig(cmdline=cmdline, data=kwargs)
    model_globstr = config['model_globstr']
    test_dataset = config['test_dataset']
    draw_curves = config['draw_curves']
    draw_heatmaps = config['draw_heatmaps']

    if model_globstr is None and test_dataset is None:
        raise ValueError('model_globstr and test_dataset are required')
        # dvc_dpath = watch.find_smart_dvc_dpath()
        # model_globstr = str(dvc_dpath / 'models/fusion/SC-20201117/*/*.pt')
        # test_dataset = dvc_dpath / 'Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json'
        # # hack for train set
        # # test_dataset = dvc_dpath / 'Drop1-Aligned-L1/combo_train_US_R001_small_nowv.kwcoco.json'
        # gpus = 'auto'

    dvc_dpath = watch.find_smart_dvc_dpath()

    HISTORICAL_MODELS_OF_INTEREST = [
        # 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v30/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v30_epoch=29-step=1284389.pt',
        dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v30/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v30_epoch=29-step=1284389.pt',
        dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_raw_v39/SC_smt_it_stm_p8_newanns_weighted_raw_v39_epoch=52-step=2269088.pt',
        dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_centerannot_raw_v42/SC_smt_it_stm_p8_centerannot_raw_v42_epoch=5-step=89465.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_smt_it_stm_p8_L1_raw_v53/BAS_smt_it_stm_p8_L1_raw_v53_epoch=15-step=340047.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BOTH_smt_it_stm_p8_L1_DIL_v55/BOTH_smt_it_stm_p8_L1_DIL_v55_epoch=5-step=53819.pt',
    ]

    HARDCODED = list(map(ub.Path, [
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=1-step=17305.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=4-step=43264.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=5-step=51917.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_ALL_REGIONS_v084/BAS_TA1_ALL_REGIONS_v084_epoch=30-step=268242.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_KOREA_v083/BAS_TA1_KOREA_v083_epoch=3-step=7459.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_KOREA_v083/BAS_TA1_KOREA_v083_epoch=4-step=9324.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_KOREA_v083/BAS_TA1_KOREA_v083_epoch=5-step=11189.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_c001_v076/BAS_TA1_c001_v076_epoch=90-step=186367.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_c001_v076/BAS_TA1_c001_v076_epoch=12-step=26623.pt',
        dvc_dpath / 'models/fusion/SC-20201117/BAS_TA1_c001_v082/BAS_TA1_c001_v082_epoch=42-step=88063.pt',
    ]))

    # with_saliency = 'auto'
    # with_class = 'auto'
    with_saliency = 'auto'
    with_class = 'auto'

    workers_per_queue = 4
    recompute = False

    # HARD CODED
    # model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    # test_dataset_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_combo11.kwcoco.json'

    # model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    # model_dpath = dvc_dpath / 'models/fusion/SC-20201117'
    # test_dataset_fpath = dvc_dpath / 'Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json'
    test_dataset_fpath = ub.Path(test_dataset)
    assert test_dataset_fpath.exists()

    annotations_dpath = config['annotations_dpath']
    if annotations_dpath is None:
        annotations_dpath = dvc_dpath / 'annotations'

    def package_metadata(package_fpath):
        # Hack for choosing one model from this "type"
        try:
            epoch_num = int(package_fpath.name.split('epoch=')[1].split('-')[0])
            expt_name = package_fpath.name.split('_epoch')[0]
        except Exception:
            # Try to read package metadata
            pkg_zip = ub.zopen(package_fpath, ext='.pt')
            found = None
            for member in pkg_zip.namelist():
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

        info = {
            'name': expt_name,
            'epoch': epoch_num,
            'fpath': package_fpath,
        }
        return info

    packages_to_eval = []
    import glob
    if model_globstr == 'special:HISTORY':
        for package_fpath in HISTORICAL_MODELS_OF_INTEREST:
            assert package_fpath.exists()
            package_info = package_metadata(ub.Path(package_fpath))
            packages_to_eval.append(package_info)
    if model_globstr == 'special:HARDCODED':
        for package_fpath in HARDCODED:
            assert package_fpath.exists(), f'{package_fpath}'
            package_info = package_metadata(ub.Path(package_fpath))
            packages_to_eval.append(package_info)
    else:
        for package_fpath in glob.glob(model_globstr, recursive=True):
            package_info = package_metadata(ub.Path(package_fpath))
            packages_to_eval.append(package_info)

    print(f'{len(packages_to_eval)=}')

    queue_dpath = dvc_dpath / '_cmd_queue_schedule'
    queue_dpath.mkdir(exist_ok=True)

    gpus = config['gpus']
    print('gpus = {!r}'.format(gpus))
    if gpus == 'auto':
        # Use all unused gpus
        import netharn as nh
        GPUS = []
        for gpu_idx, gpu_info in nh.device.gpu_info().items():
            print('gpu_idx = {!r}'.format(gpu_idx))
            print('gpu_info = {!r}'.format(gpu_info))
            if len(gpu_info['procs']) == 0:
                GPUS.append(gpu_idx)
    else:
        GPUS = gpus

    print('GPUS = {!r}'.format(GPUS))
    environ = {
        'DVC_DPATH': dvc_dpath,
    }

    from watch.utils import cmd_queue
    queue = cmd_queue.Queue.create(config['backend'], name='schedule-eval',
                                   size=len(GPUS), environ=environ,
                                   dpath=queue_dpath, gres=GPUS)

    virtualenv_cmd = config['virtualenv_cmd']
    if virtualenv_cmd:
        queue.add_header_command(virtualenv_cmd)

    def ensure_iterable(x):
        return x if ub.iterable(x) else [x]

    pred_cfg_basis = {}
    pred_cfg_basis['tta_time'] = ensure_iterable(config['tta_time'])
    pred_cfg_basis['tta_fliprot'] = ensure_iterable(config['tta_fliprot'])
    pred_cfg_basis['chip_overlap'] = ensure_iterable(config['chip_overlap'])

    expanded_packages_to_eval = []
    for raw_info in packages_to_eval:
        for pred_cfg in ub.named_product(pred_cfg_basis):
            info = raw_info.copy()
            package_fpath = info['fpath']

            # Hack for defaults so they keep the same hash
            # Can remove this once current phase is done
            if 1:
                pred_cfg_hack = pred_cfg.copy()
                if pred_cfg['tta_fliprot'] is None:
                    pred_cfg_hack.pop('tta_fliprot')
                if pred_cfg['tta_time'] is None:
                    pred_cfg_hack.pop('tta_time')
                if pred_cfg['chip_overlap'] == 0.3:
                    pred_cfg_hack.pop('chip_overlap')

            suggestions = organize.suggest_paths(
                package_fpath=package_fpath,
                test_dataset=test_dataset_fpath,
                sidecar2=config['sidecar2'], as_json=False,
                workdir=config['workdir'],
                pred_cfg=pred_cfg_hack,
            )
            info['suggestions'] = suggestions
            info['pred_cfg'] = pred_cfg
            expanded_packages_to_eval.append(info)

    skip_existing = config['skip_existing']

    with_pred = config['enable_pred']  # TODO: allow caching
    with_eval = config['enable_eval']
    with_track = config['enable_track']
    with_iarpa_eval = config['enable_iarpa_eval']

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

    def lazy_command(stamp_fpath, command):
        """
        Augments the command so it is lazy if its output exists

        TODO: incorporate into cmdq
        """
        if stamp_fpath is not None:
            command = f'test -f "{stamp_fpath}" || \\\n  ' + command
        return command

    if with_eval == 'redo':
        # Need to dvc unprotect
        needs_unprotect = []
        for info in expanded_packages_to_eval:
            suggestions = info['suggestions']
            pred_dataset_fpath = ub.Path(suggestions['pred_dataset'])  # NOQA
            eval_metrics_fpath = ub.Path(suggestions['eval_dpath']) / 'curves/measures2.json'
            eval_metrics_dvc_fpath = ub.Path(suggestions['eval_dpath']) / 'curves/measures2.json.dvc'

            if eval_metrics_dvc_fpath.exists():
                needs_unprotect.append(eval_metrics_fpath)

        if needs_unprotect:
            from watch.utils.simple_dvc import SimpleDVC
            simple_dvc = SimpleDVC(dvc_dpath)
            simple_dvc.unprotect(needs_unprotect)

    if config['shuffle_jobs']:
        import kwarray
        expanded_packages_to_eval = kwarray.shuffle(expanded_packages_to_eval)

    for info in expanded_packages_to_eval:
        package_fpath = info['fpath']
        suggestions = info['suggestions']
        pred_cfg = info['pred_cfg']
        pred_dataset_fpath = ub.Path(suggestions['pred_dataset'])  # NOQA
        eval_metrics_fpath = ub.Path(suggestions['eval_dpath']) / 'curves/measures2.json'
        eval_metrics_dvc_fpath = ub.Path(suggestions['eval_dpath']) / 'curves/measures2.json.dvc'

        suggestions['eval_metrics'] = eval_metrics_fpath
        suggestions['test_dataset'] = test_dataset_fpath
        suggestions['true_dataset'] = test_dataset_fpath
        suggestions['package_fpath'] = package_fpath
        suggestions['with_class'] = with_class
        suggestions['with_saliency'] = with_saliency
        # suggestions = ub.map_vals(lambda x: str(x).replace(
        #     str(dvc_dpath), '$DVC_DPATH'), suggestions)
        predictkw = {
            'workers_per_queue': workers_per_queue,
            **pred_cfg,
        }

        # We can skip evaluation if we already have a metrics dvc file
        # (even if we haven't pulled those actual metrics to disk)
        has_eval = eval_metrics_dvc_fpath.exists() or eval_metrics_fpath.exists()
        has_pred = pred_dataset_fpath.exists()

        task_infos = {
            'pred': {
                'enabled': with_pred or recompute_pred,
                'output': pred_dataset_fpath,
                'requires': [],
                'recompute': recompute_pred,
            },
            'eval': {
                'enabled': with_eval or recompute_eval,
                'output': eval_metrics_fpath,
                'requires': ['pred'],
                'recompute': recompute_eval,
            }
        }

        def should_compute_task(task_info):
            # Check if each dependency will exist by the time we run this job
            deps_will_exist = []
            for req in task_info['requires']:
                req_info = task_infos[req]
                # If the req is not enabled, then it output must exist now.
                will_exist = req_info['enabled'] or req_info['output'].exists()
                deps_will_exist.append(will_exist)

            # can only eval predictions that exist or will exist
            return all(deps_will_exist) and task_info['enabled']

        name_suffix = (
            '_' + ub.hash_data(str(package_fpath))[0:8] +
            '_' + ub.hash_data(pred_cfg)[0:8]
        )

        task = 'pred'
        pred_job = None
        task_info = task_infos[task]
        if should_compute_task(task_info):
            command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.predict \
                    --write_probs=True \
                    --write_preds=False \
                    --with_class={with_class} \
                    --with_saliency={with_saliency} \
                    --with_change=False \
                    --package_fpath={package_fpath} \
                    --pred_dataset={pred_dataset} \
                    --test_dataset={test_dataset} \
                    --num_workers={workers_per_queue} \
                    --chip_overlap={chip_overlap} \
                    --tta_time={tta_time} \
                    --tta_fliprot={tta_fliprot} \
                    --gpus=0, \
                    --batch_size=1
                ''').format(**suggestions, **predictkw)
            if not task_info['recompute']:
                # Only run the command if its expected output does not exist
                command = lazy_command(task_info['output'], command)

            if task_info['recompute']:
                needs_compute = True
            else:
                needs_compute = not (skip_existing and (has_pred or has_eval))

            if needs_compute:
                name = 'pred' + name_suffix
                pred_cpus = workers_per_queue
                pred_job = queue.submit(command, gpus=1, name=name,
                                        cpus=pred_cpus,
                                        partition=config['partition'],
                                        mem=config['mem'])

        task = 'eval'
        eval_job = None
        task_info = task_infos[task]
        if should_compute_task(task_info):
            command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.evaluate \
                    --true_dataset={true_dataset} \
                    --pred_dataset={pred_dataset} \
                      --eval_dpath={eval_dpath} \
                      --score_space=video \
                      --draw_curves={draw_curves} \
                      --draw_heatmaps={draw_heatmaps} \
                      --workers=2
                ''').format(**suggestions, draw_curves=draw_curves,
                            draw_heatmaps=draw_heatmaps)

            if not task_info['recompute']:
                # Only run the command if its expected output does not exist
                command = lazy_command(task_info['output'], command)

            if task_info['recompute']:
                needs_compute = True
            else:
                needs_compute = not (skip_existing and has_eval)

            if needs_compute:
                name = 'eval' + name_suffix
                eval_job = queue.submit(
                    command, depends=pred_job, name=name, cpus=2,
                    partition=config['partition'], mem=config['mem'])
            task_info['job'] = eval_job

        tracking_param_basis = {
            'thresh': [0.1],
            # 'thresh': [0.1, 0.2, 0.3],
        }
        for track_cfg in ub.named_product(tracking_param_basis):
            from watch.tasks.fusion import schedule_iarpa_eval
            track_suggestions = schedule_iarpa_eval._suggest_track_paths(
                pred_dataset_fpath, track_cfg)
            name_suffix = '-'.join([
                'pkg', suggestions['package_cfgstr'],
                'prd', suggestions['pred_cfgstr'],
                'trk', track_suggestions['track_cfgstr'],
            ])
            iarpa_eval_dpath = track_suggestions['iarpa_eval_dpath']
            track_out_fpath = track_suggestions['track_out_fpath']
            iarpa_summary_fpath = track_suggestions['iarpa_summary_fpath']

            task_infos.update({
                'track': {
                    'enabled': with_track or recompute_track,
                    'output': track_out_fpath,
                    'requires': ['pred'],
                    'recompute': recompute_track,
                },
                'iarpa_eval': {
                    'enabled': with_iarpa_eval or recompute_iarpa_eval,
                    'output': iarpa_summary_fpath,
                    'requires': ['track'],
                    'recompute': recompute_iarpa_eval,
                }
            })

            task = 'track'
            track_job = None
            task_info = task_infos[task]
            if should_compute_task(task_info):

                track_info = schedule_iarpa_eval._build_bas_track_job(
                    pred_dataset_fpath, track_out_fpath, **track_cfg)
                command = track_info['command']

                if not task_info['recompute']:
                    # Only run the command if its expected output does not exist
                    command = lazy_command(task_info['output'], command)

                if task_info['recompute']:
                    needs_compute = True
                else:
                    needs_compute = not (skip_existing and has_pred)

                if needs_compute:
                    name = 'track-' + name_suffix
                    track_job = queue.submit(
                        command=command,
                        depends=pred_job,
                        name=name,
                        cpus=2,
                        partition=config['partition'],
                        mem=config['mem'],
                    )
                    task_info['job'] = track_job

            # TODO: need a way of knowing if a package is BAS or SC.
            # Might need info on GSD as well.
            task = 'iarpa_eval'
            iarpa_eval_job = None
            task_info = task_infos[task]
            if should_compute_task(task_info):
                iarpa_eval_info = schedule_iarpa_eval._build_iarpa_eval_job(
                    track_out_fpath, iarpa_eval_dpath, annotations_dpath, name_suffix)

                command = iarpa_eval_info['command']

                if not task_info['recompute']:
                    # Only run the command if its expected output does not exist
                    command = lazy_command(task_info['output'], command)

                if task_info['recompute']:
                    needs_compute = True
                else:
                    needs_compute = not (skip_existing and has_pred)

                if needs_compute:
                    name = 'iarpaeval-' + name_suffix
                    iarpa_eval_job = queue.submit(
                        command=command,
                        depends=track_job,
                        name=name,
                        cpus=2,
                        partition=config['partition'],
                        mem=config['mem'],
                    )
                    task_info['job'] = iarpa_eval_job

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

    """
    # Now postprocess script:

    python ~/code/watch/watch/tasks/fusion/organize.py make_eval_symlinks

    ls models/fusion/unevaluated-activity-2021-11-12/eval_links

    cd /home/joncrall/remote/horologic/smart_watch_dvc

    ARR=($(ls -a1 models/fusion/unevaluated-activity-2021-11-12/eval_links/*/curves/measures2.json))
    for ARG in "${ARR[@]}"; do
        echo "ARG = $ARG"
        cat "$ARG" | jq '.ovr_measures[] | with_entries(select(.key | in({"node":1, "ap":1, "auc": 1})))'
    done
    print('MEASURE_FPATHS = {!r}'.format(MEASURE_FPATHS))
    feh models/fusion/unevaluated-activity-2021-11-12/eval_links/*/curves/ovr_roc.png
    feh models/fusion/unevaluated-activity-2021-11-12/eval_links/*/curves/ovr_ap.png
    """


def updates_dvc_measures():
    """
    DEPRECATE

    Add results of pixel evaluations to DVC
    """
    import watch
    # import functools
    import os
    dvc_dpath = watch.find_smart_dvc_dpath()
    dpath = dvc_dpath / 'models/fusion/SC-20201117'
    measures_fpaths = list(dpath.glob('*/*/*/eval/curves/measures2.json'))

    is_symlink = ub.memoize(os.path.islink)
    # is_symlink = functools.cache(os.path.islink)
    # import timerit
    # ti = timerit.Timerit(100, bestof=10, verbose=2)
    # for timer in ti.reset('time'):
    #     with timer:
    #         is_symlink(fpath))

    def check_if_contained_in_symlink(fpath, dvc_dpath):
        rel_fpath = fpath.relative_to(dvc_dpath)
        parts = rel_fpath.parts
        curr = fpath.parent
        for i in range(len(parts)):
            if is_symlink(curr):
                return True
            curr = curr.parent

    needs_add = []
    for fpath in measures_fpaths:
        dvc_fpath = ub.Path(str(fpath) + '.dvc')
        if not dvc_fpath.exists():
            if not fpath.is_symlink():
                if not check_if_contained_in_symlink(fpath, dvc_dpath):
                    rel_fpath = fpath.relative_to(dvc_dpath)
                    needs_add.append(rel_fpath)

    print(f'Need to add {len(needs_add)} summaries')
    rel_fpaths = [str(p) for p in needs_add]

    import os
    import dvc.main
    push_dpath = '/'.join(os.path.commonprefix([
        ub.Path(p).parts for p in rel_fpaths]))
    # from dvc import main
    saved_cwd = os.getcwd()
    try:
        os.chdir(dvc_dpath)
        dvc_command = ['add'] + rel_fpaths
        dvc.main.main(dvc_command)

        remote = 'horologic'
        dvc_command = ['push', '-r', remote, '--recursive', str(push_dpath)]
        dvc.main.main(dvc_command)
    finally:
        os.chdir(saved_cwd)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/schedule_evaluation.py schedule_evaluation

        python ~/code/watch/watch/tasks/fusion/organize.py make_nice_dirs
        python ~/code/watch/watch/tasks/fusion/organize.py make_eval_symlinks
        python ~/code/watch/watch/tasks/fusion/organize.py make_pred_symlinks
        python ~/code/watch/watch/tasks/fusion/schedule_evaluation.py
    """
    schedule_evaluation(cmdline=True)

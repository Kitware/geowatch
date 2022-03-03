"""
Helper for scheduling a set of prediction + evaluation jobs

python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation
"""
import ubelt as ub
from watch.utils import tmux_queue


def schedule_evaluation(model_globstr=None, test_dataset=None, gpus='auto',
                        run=False, with_rich=0, with_status=True,
                        virtualenv_cmd=None, skip_existing=False):
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
    import json

    if model_globstr is None and test_dataset is None:
        dvc_dpath = watch.find_smart_dvc_dpath()
        model_globstr = str(dvc_dpath / 'models/fusion/SC-20201117/*/*.pt')
        test_dataset = dvc_dpath / 'Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json'

        # hack for train set
        # test_dataset = dvc_dpath / 'Drop1-Aligned-L1/combo_train_US_R001_small_nowv.kwcoco.json'
        gpus = 'auto'

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

    with_pred = True  # TODO: allow caching
    with_eval = True
    workers_per_queue = 5
    recompute = False

    # HARD CODED
    # model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    # test_dataset_fpath = dvc_dpath / 'Drop1-Aligned-L1/vali_combo11.kwcoco.json'

    # model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    # model_dpath = dvc_dpath / 'models/fusion/SC-20201117'
    # test_dataset_fpath = dvc_dpath / 'Drop1-Aligned-L1/combo_vali_nowv.kwcoco.json'
    test_dataset_fpath = ub.Path(test_dataset)
    assert test_dataset_fpath.exists()

    stamp = ub.timestamp() + '_' + ub.hash_data([])[0:8]

    def package_metadata(package_fpath):
        # Hack for choosing one model from this "type"
        import xdev
        with xdev.embed_on_exception_context:
            epoch_num = int(package_fpath.name.split('epoch=')[1].split('-')[0])
        expt_name = package_fpath.name.split('_epoch')[0]
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

    shuffle_jobs = True
    if shuffle_jobs:
        import kwarray
        packages_to_eval = kwarray.shuffle(packages_to_eval)

    print(f'{len(packages_to_eval)=}')

    # # for subfolder in model_dpath.glob('*'):
    #     # package_fpaths = list(subfolder.glob('*.pt'))
    #     subfolder_infos = [package_metadata(package_fpath)
    #                        for package_fpath in package_fpaths]
    #     subfolder_infos = sorted(subfolder_infos, key=lambda x: x['epoch'], reverse=True)
    #     for info in subfolder_infos:
    #         if 'rutgers_v5' in info['name']:
    #             break
    #         packages_to_eval.append(info)
    #         # break

    tmux_schedule_dpath = dvc_dpath / '_tmp_tmux_schedule'
    tmux_schedule_dpath.mkdir(exist_ok=True)

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
    # GPUS = [0, 1, 2, 3]
    # GPUS = [0]
    environ = {
        'DVC_DPATH': dvc_dpath,
    }

    tq = tmux_queue.TMUXMultiQueue(
        name=stamp, size=len(GPUS), environ=environ, gres=GPUS,
        dpath=tmux_schedule_dpath)
    if virtualenv_cmd:
        tq.add_header_command(virtualenv_cmd)

    recompute_pred = recompute
    recompute_eval = recompute or 0

    for info, queue in zip(packages_to_eval, tq):
        package_fpath = info['fpath']
        suggestions = organize.suggest_paths(package_fpath=package_fpath, test_dataset=test_dataset_fpath)
        suggestions = json.loads(suggestions)

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
        }

        print('pred_dataset_fpath = {!r}'.format(pred_dataset_fpath))
        has_eval = eval_metrics_dvc_fpath.exists() or eval_metrics_fpath.exists()
        has_pred = pred_dataset_fpath.exists()
        print('has_eval = {!r}'.format(has_eval))
        print('has_pred = {!r}'.format(has_pred))

        if with_pred:
            pred_command = ub.codeblock(
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
                    --compress=DEFLATE \
                    --gpus=0, \
                    --batch_size=1
                ''').format(**suggestions, **predictkw)
            if not recompute_pred:
                # Only run the command if its expected output does not exist
                pred_command = (
                    '[[ -f "{pred_dataset}" ]] || '.format(**suggestions) +
                    pred_command
                )

            if recompute_pred or not (skip_existing and (has_pred or has_eval)):
                if not has_eval:
                    queue.submit(pred_command)

        if with_eval:
            eval_command = ub.codeblock(
                r'''
                python -m watch.tasks.fusion.evaluate \
                    --true_dataset={true_dataset} \
                    --pred_dataset={pred_dataset} \
                      --eval_dpath={eval_dpath} \
                      --score_space=video \
                      --draw_curves=1 \
                      --draw_heatmaps=1
                ''').format(**suggestions)
            if not recompute_eval:
                # TODO: use a real stamp file
                # Only run the command if its expected output does not exist
                eval_command = (
                    '[[ -f "{eval_metrics}" ]] || '.format(**suggestions) +
                    eval_command
                )
            if recompute_eval or not (skip_existing and has_eval):
                queue.submit(eval_command)

    print('tq = {!r}'.format(tq))
    # print(f'{len(tq)=}')
    tq.rprint(with_status=with_status, with_rich=with_rich)

    # RUN
    if run:
        # ub.cmd('bash ' + str(driver_fpath), verbose=3, check=True)
        tq.run()
        agg_state = tq.monitor()
        if not agg_state['errored']:
            tq.kill()
    else:
        driver_fpath = tq.write()
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

        python ~/code/watch/watch/tasks/fusion/schedule_evaluation.py gather_measures
    """
    import fire
    fire.Fire()

import scriptconfig as scfg
import ubelt as ub


class TeamFeaturePipelineConfig(scfg.Config):
    """
    This generates the bash commands necessary to run team feature computation,
    followed by aggregation and then splitting out train / val datasets.

    Note:
        The models and parameters to use are hard coded in this script.
    """
    default = {
        'base_fpath': scfg.Value('auto', help=ub.paragraph(
            '''
            base coco file to compute team-features on, combine, and split. If
            auto, uses a hard-coded value
            ''')),
        'dvc_dpath': scfg.Value('auto', help=ub.paragraph(
            '''
            The DVC directory where team feature model weights can be found.
            If "auto" uses the ``watch.find_smart_dvc_dpath`` mechanism
            to infer the location.
            ''')),
        'gres': scfg.Value('auto', help='comma separated list of gpus or auto'),

        'with_landcover': scfg.Value(True, help='Include DZYNE landcover features'),
        'with_materials': scfg.Value(True, help='Include Rutgers material features'),
        'with_invariants': scfg.Value(True, help='Include UKY invariant features'),
        'with_depth': scfg.Value(True, help='Include DZYNE WorldView depth features'),

        'virtualenv_cmd': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your bashrc
            does not start it by default.''')),

        'data_workers': scfg.Value(2, help='dataloader workers for each proc'),
        'keep_sessions': scfg.Value(False, help='if True does not close tmux sessions'),

        'workers': scfg.Value('auto', help='Maximum number of parallel jobs, 0 is no-nonsense serial mode. '),
        'run': scfg.Value(True, help='if True execute the pipeline'),
        'cache': scfg.Value(True, help='if True skip completed results'),

        'follow': scfg.Value(False),
    }


def main(cmdline=True, **kwargs):
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.

    TODO:
        - [ ] Option to just dump the serial bash script that does everything.
    """
    from watch.utils import tmux_queue
    from watch.utils.lightning_ext import util_globals
    from scriptconfig.smartcast import smartcast

    config = TeamFeaturePipelineConfig(cmdline=cmdline, data=kwargs)

    gres = config['gres']
    gres = smartcast(gres)
    print('gres = {!r}'.format(gres))
    if gres is None:
        import netharn as nh
        gres = []
        for gpu_idx, gpu_info in nh.device.gpu_info().items():
            if len(gpu_info['procs']) == 0:
                gres.append(gpu_idx)
    elif not ub.iterable(gres):
        gres = [gres]

    workers = config['workers']
    if workers == 'auto':
        if gres is None:
            workers = 0
        else:
            workers = len(gres)
    data_workers = util_globals.coerce_num_workers(config['data_workers'])

    if config['dvc_dpath'] == 'auto':
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
    else:
        dvc_dpath = ub.Path(config['dvc_dpath'])

    if config['base_fpath'] == 'auto':
        # Auto hack.
        base_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
        # base_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
    else:
        base_fpath = ub.Path(config['base_fpath'])

    aligned_bundle_dpath = base_fpath.parent

    model_fpaths = {
        'rutgers_materials': dvc_dpath / 'models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth',
        'dzyne_landcover': dvc_dpath / 'models/landcover/visnav_remap_s2_subset.pt',
        'uky_pretext': dvc_dpath / 'models/uky/uky_invariants_2022_01/pretext/pretext.ckpt',
        'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_01/segmentation/segmentation.ckpt',
        'dzyne_depth': dvc_dpath / 'models/depth/weights_v1.pt',
    }

    outputs = {
        'rutgers_materials': aligned_bundle_dpath / 'rutgers_mat_seg.kwcoco.json',
        'dzyne_landcover': aligned_bundle_dpath / 'dzyne_landcover.kwcoco.json',
        'dzyne_depth': aligned_bundle_dpath / 'dzyne_depth.kwcoco.json',
        'uky_invariants': aligned_bundle_dpath / 'uky_invariants.kwcoco.json',
    }

    codes = {
        'with_landcover': 'L',
        'with_depth': 'D',
        'with_materials': 'M',
        'with_invariants': 'I',
    }

    tasks = []
    # tmux queue is still limited. The order of submission matters.

    combo_code_parts = []
    key = 'with_landcover'
    if config[key]:
        # Landcover is fairly fast to run, do it first
        task = {}
        task['output_fpath'] = outputs['dzyne_landcover']
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.landcover.predict \
                --dataset="{base_fpath}" \
                --deployed="{model_fpaths['dzyne_landcover']}" \
                --output="{task['output_fpath']}" \
                --num_workers="{data_workers}" \
                --device=0
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    key = 'with_depth'
    if config[key]:
        # Landcover is fairly fast to run, do it first
        task = {}
        # Only need 1 worker to minimize lag between images, task is GPU bound
        depth_data_workers = max(2, data_workers)
        depth_window_size = 1536  # takes 18GB
        task['output_fpath'] = outputs['dzyne_depth']
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.depth.predict \
                --dataset="{base_fpath}" \
                --output="{task['output_fpath']}" \
                --deployed="{model_fpaths['dzyne_depth']}" \
                --data_workers={depth_data_workers} \
                --window_size={depth_window_size}
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    # Run materials while landcover is running
    key = 'with_materials'
    if config[key]:
        task = {}
        task['output_fpath'] = outputs['rutgers_materials']
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.rutgers_material_seg.predict \
                --test_dataset="{base_fpath}" \
                --checkpoint_fpath="{model_fpaths['rutgers_materials']}" \
                --pred_dataset="{task['output_fpath']}" \
                --default_config_key=iarpa \
                --num_workers="{data_workers}" \
                --batch_size=32 --gpus "0" \
                --compress=DEFLATE --blocksize=64
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    # When landcover finishes run invariants
    # Note: Does not run on a 1080, needs 18GB in this form
    key = 'with_invariants'
    if config[key]:
        task = {}
        task['output_fpath'] = outputs['uky_invariants']
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.invariants.predict \
                --input_kwcoco "{base_fpath}" \
                --output_kwcoco "{task['output_fpath']}" \
                --pretext_ckpt_path "{model_fpaths['uky_pretext']}" \
                --segmentation_ckpt "{model_fpaths['uky_segmentation']}" \
                --do_pca 1 \
                --num_dim 8 \
                --num_workers="{data_workers}" \
                --write_workers 2 \
                --tasks all
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    if workers == 0:
        gres = None

    if gres is None:
        size = min(len(tasks), max(1, workers))
    else:
        size = min(len(tasks), len(gres))

    tq = tmux_queue.TMUXMultiQueue(name='teamfeat', size=size, gres=gres)
    if config['virtualenv_cmd']:
        tq.add_header_command(config['virtualenv_cmd'])

    for task in tasks:
        if config['cache']:
            if not task['output_fpath'].exists():
                command = f"[[ -f '{task['output_fpath']}' ]] || " + task['command']
                tq.submit(command)
        else:
            tq.submit(task['command'])

    if workers > 0:
        # Launch this TQ if there are parallel workers, otherwise just make a
        # longer serial script
        tq.rprint()
        tq.write()

        # TODO: make the monitor spawn in a new tmux session. The monitor could
        # actually be the scheduler process.
        if config['run']:
            import subprocess
            try:
                agg_state = tq.run(block=True)
            except subprocess.CalledProcessError as ex:
                print('ex.stdout = {!r}'.format(ex.stdout))
                print('ex.stderr = {!r}'.format(ex.stderr))
                print('ex.returncode = {!r}'.format(ex.returncode))
                raise
            else:
                if not config['keep_sessions']:
                    if not agg_state['errored']:
                        tq.kill()

        tq = tmux_queue.TMUXMultiQueue(name='combine-feats', size=2)

    if 1:
        # Finalize features by combining them all into combo.kwcoco.json
        tocombine = [str(base_fpath)] + [str(task['output_fpath']) for task in tasks]
        combo_code = ''.join(sorted(combo_code_parts))

        base_combo_fpath = aligned_bundle_dpath / f'combo_{combo_code}.kwcoco.json'

        if config['virtualenv_cmd']:
            tq.add_header_command(config['virtualenv_cmd'])

        # TODO: enable forcing if needbe
        if not base_combo_fpath.exists() or not config['cache']:
            #  Indent of this the codeblock matters for this line
            src_lines = ' \\\n                          '.join(tocombine)
            command = ub.codeblock(
                fr'''
                python -m watch.cli.coco_combine_features \
                    --src {src_lines} \
                    --dst {base_combo_fpath}
                ''')
            tq.submit(command)

        tq.rprint()

        if config['run']:
            follow = config['follow']
            if follow and workers == 0 and len(tq.workers) == 1:
                queue = tq.workers[0]
                fpath = queue.write()
                ub.cmd(f'bash {fpath}', verbose=3, check=True)
            else:
                agg_state = tq.run(block=True)
                if config['follow']:
                    tq.monitor()
            if not config['keep_sessions']:
                if not agg_state['errored']:
                    tq.kill()

            # Also call the prepare-splits script
            from watch.cli import prepare_splits
            split_config = ub.dict_isect(
                config, prepare_splits.PrepareSplitsConfig.default)
            split_config['base_fpath'] = base_combo_fpath
            prepare_splits.main()

    """
    Ignore:
        python -m kwcoco stats data.kwcoco.json uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
        python -m watch stats uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
    """


if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        python -m watch.cli.prepare_teamfeats \
            --base_fpath="$DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json" \
            --gres=0 \
            --with_depth=True \
            --keep_sessions=False \
            --run=False --cache=False --virtualenv_cmd "conda activate watch"

        python -m watch.cli.prepare_teamfeats --gres=0,2 --with_depth=True --keep_sessions=True
        python -m watch.cli.prepare_teamfeats --gres=2 --with_materials=False --keep_sessions=True

        # TODO: rename to schedule teamfeatures

        # TO UPDATE ANNOTS
        # Update to whatever the state of the annotations submodule is
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        python -m watch project_annotations \
            --src $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json \
            --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data_20220203.kwcoco.json \
            --viz_dpath $DVC_DPATH/Drop2-Aligned-TA1-2022-01/_viz_propogate \
            --site_models="$DVC_DPATH/annotations/site_models/*.geojson"

        kwcoco stats $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data_20220203.kwcoco.json $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json

        # Team Features on Drop2
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/data_20220203.kwcoco.json \
            --gres=0,1 --with_depth=True --with_materials=False --keep_sessions=True --run=1

    """
    main(cmdline=True)

import scriptconfig as scfg
import ubelt as ub


class TeamFeaturePipelineConfig(scfg.Config):
    default = {
        'dvc_dpath': scfg.Value('auto'),
        'bundle_name': 'Drop1-Aligned-L1-2022-01',
        'base_coco_name': 'data.kwcoco.json',
        'gres': scfg.Value('auto', help='comma separated list of gpus or auto'),

        'with_landcover': True,
        'with_materials': True,
        'with_invariants': True,
        'with_depth': False,

        'data_workers': scfg.Value(2, help='dataloader workers for each proc'),
        'keep_sessions': scfg.Value(False, help='if True does not close tmux sessions'),
    }


def main(cmdline=True, **kwargs):
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.
    """
    import watch
    from watch.utils import tmux_queue
    from watch.utils.lightning_ext import util_globals
    from scriptconfig.smartcast import smartcast

    config = TeamFeaturePipelineConfig(cmdline=cmdline, data=kwargs)

    data_workers = util_globals.coerce_num_workers(config['data_workers'])

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

    if config['dvc_dpath'] == 'auto':
        config['dvc_dpath'] = watch.find_smart_dvc_dpath()

    dvc_dpath = ub.Path(config['dvc_dpath'])
    aligned_bundle_dpath = dvc_dpath / config['bundle_name']
    base_coco_fpath = aligned_bundle_dpath / config['base_coco_name']

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
                --dataset="{base_coco_fpath}" \
                --deployed="{model_fpaths['dzyne_landcover']}" \
                --output="{task['output_fpath']}" \
                --num_workers="{data_workers}" \
                --device=0 \
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    key = 'with_depth'
    if config[key]:
        # Landcover is fairly fast to run, do it first
        task = {}
        task['output_fpath'] = outputs['dzyne_landcover']
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.depth.predict \
                --dataset="{base_coco_fpath}" \
                --output="{task['output_fpath']}" \
                --deployed="{model_fpaths['dzyne_landcover']}" \
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
                --test_dataset="{base_coco_fpath}" \
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
                --input_kwcoco "{base_coco_fpath}" \
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

    # gres = [0, 1]
    size = min(len(tasks), len(gres))
    tq = tmux_queue.TMUXMultiQueue(name='teamfeat', size=size, gres=gres)

    for task in tasks:
        if not task['output_fpath'].exists():
            command = f"[[ -f '{task['output_fpath']}' ]] || " + task['command']
            tq.submit(command)

    tq.rprint()

    tq.write()

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

    if 1:
        # Finalize features by combining them all into combo.kwcoco.json
        tocombine = [str(base_coco_fpath)] + [str(task['output_fpath']) for task in tasks]
        combo_code = ''.join(sorted(combo_code_parts))
        combo_fpath = aligned_bundle_dpath / f'combo_{combo_code}.kwcoco.json'

        # TODO: enable forcing if needbe
        if not combo_fpath.exists() or 0:
            command = ub.codeblock(
                f'''
                python -m watch.cli.coco_combine_features \
                    --src {' '.join(tocombine)} \
                    --dst {combo_fpath}
                ''')
            ub.cmd(command, verbose=2, check=True)

        splits = {
            'combo_train': combo_fpath.augment(suffix='_train'),
            'combo_nowv_train': combo_fpath.augment(suffix='_nowv_train'),
            'combo_wv_train': combo_fpath.augment(suffix='_wv_train'),

            'combo_vali': combo_fpath.augment(suffix='_vali'),
            'combo_nowv_vali': combo_fpath.augment(suffix='_nowv_vali'),
            'combo_wv_vali': combo_fpath.augment(suffix='_wv_vali'),
        }

        tq = tmux_queue.TMUXMultiQueue(name='watch-splits', size=2)

        # Perform train/validation splits with and without worldview
        command = ub.codeblock(
            f'''
            python -m kwcoco subset \
                --src {combo_fpath} \
                --dst {splits['combo_train']} \
                --select_videos '.name | startswith("KR_") | not'
            ''')
        tq.submit(command, index=0)

        command = ub.codeblock(
            f'''
            python -m kwcoco subset \
                --src {splits['combo_train']} \
                --dst {splits['combo_nowv_train']} \
                --select_images '.sensor_coarse != "WV"'
            ''')
        tq.submit(command, index=0)

        command = ub.codeblock(
            f'''
            python -m kwcoco subset \
                --src {splits['combo_train']} \
                --dst {splits['combo_wv_train']} \
                --select_images '.sensor_coarse == "WV"'
            ''')
        tq.submit(command, index=0)

        # Perform vali/validation splits with and without worldview
        command = ub.codeblock(
            f'''
            python -m kwcoco subset \
                --src {combo_fpath} \
                --dst {splits['combo_vali']} \
                --select_videos '.name | startswith("KR_")'
            ''')
        tq.submit(command, index=1)

        command = ub.codeblock(
            f'''
            python -m kwcoco subset \
                --src {splits['combo_vali']} \
                --dst {splits['combo_nowv_vali']} \
                --select_images '.sensor_coarse != "WV"'
            ''')
        tq.submit(command, index=1)

        command = ub.codeblock(
            f'''
            python -m kwcoco subset \
                --src {splits['combo_vali']} \
                --dst {splits['combo_wv_vali']} \
                --select_images '.sensor_coarse == "WV"'
            ''')
        tq.submit(command, index=1)
        agg_state = tq.run(block=True)
        if not config['keep_sessions']:
            if not agg_state['errored']:
                tq.kill()

    """
    Ignore:
        python -m kwcoco stats data.kwcoco.json uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
        python -m watch stats uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
    """


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.prepare_teamfeats --gres=2
        python -m watch.cli.prepare_teamfeats --gres=2 --with_rutgers=False
    """
    main(cmdline=True)

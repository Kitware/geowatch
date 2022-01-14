

def schedule_teamfeature_compute():
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.
    """
    import ubelt as ub
    import watch
    from watch.utils import tmux_queue

    dvc_dpath = watch.find_smart_dvc_dpath()
    aligned_bundle_dpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01'
    base_coco_fpath = aligned_bundle_dpath / 'data.kwcoco.json'

    model_fpaths = {
        'rutgers_materials': dvc_dpath / 'models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth',
        'dzyne_landcover': dvc_dpath / 'models/landcover/visnav_remap_s2_subset.pt',

        'uky_pretext': dvc_dpath / 'models/uky/uky_invariants_2022_01/pretext/pretext.ckpt',
        'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_01/segmentation/segmentation.ckpt',
    }

    outputs = {
        'rutgers_materials': aligned_bundle_dpath / 'rutgers_mat_seg.kwcoco.json',
        'dzyne_landcover': aligned_bundle_dpath / 'dzyne_landcover.kwcoco.json',
        'uky_invariants': aligned_bundle_dpath / 'uky_invariants.kwcoco.json',
    }

    tasks = []
    # tmux queue is still limited. The order of submission matters.

    # Landcover is fairly fast to run, do it first
    task = {}
    task['output_fpath'] = outputs['dzyne_landcover']
    task['command'] = ub.codeblock(
        fr'''
        python -m watch.tasks.landcover.predict \
            --dataset="{base_coco_fpath}" \
            --deployed="{model_fpaths['dzyne_landcover']}" \
            --output="{task['output_fpath']}" \
            --device=0 \
            --num_workers="8"
        ''')
    tasks.append(task)

    # Run materials while landcover is running
    task = {}
    task['output_fpath'] = outputs['rutgers_materials']
    task['command'] = ub.codeblock(
        fr'''
        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset="{base_coco_fpath}" \
            --checkpoint_fpath="{model_fpaths['rutgers_materials']}" \
            --pred_dataset="{task['output_fpath']}" \
            --default_config_key=iarpa \
            --num_workers="8" \
            --batch_size=32 --gpus "0" \
            --compress=DEFLATE --blocksize=64
        ''')
    tasks.append(task)

    # When landcover finishes run invariants
    # Note: Does not run on a 1080, needs 18GB in this form
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
            --num_workers avail/2 \
            --write_workers avail/2 \
            --tasks all
        ''')
    tasks.append(task)

    import netharn as nh
    GPUS = []
    for gpu_idx, gpu_info in nh.device.gpu_info().items():
        if len(gpu_info['procs']) == 0:
            GPUS.append(gpu_idx)

    # GPUS = [0, 1]
    tq = tmux_queue.TMUXMultiQueue(name=f'teamfeat-{ub.timestamp()}', size=len(GPUS), gres=GPUS)

    for task in tasks:
        command = f"[[ -f '{task['output_fpath']}' ]] || " + task['command']
        tq.submit(command)

    tq.rprint()

    tq.write()

    import subprocess
    try:
        tq.run()
    except subprocess.CalledProcessError as ex:
        print('ex.stdout = {!r}'.format(ex.stdout))
        print('ex.stderr = {!r}'.format(ex.stderr))
        print('ex.returncode = {!r}'.format(ex.returncode))
        raise

    if 0:
        tocombine = [str(base_coco_fpath)] + [str(task['output_fpath']) for task in tasks]
        combined_fpath = str(aligned_bundle_dpath / 'combo.kwcoco.json')
        command = ub.codeblock(
            f'''
            python -m watch.cli.coco_combine_features \
                --src {' '.join(tocombine)} \
                --dst {combined_fpath}
            ''')
        print(command)
        ub.cmd(command, verbose=2, check=True)

    tq.monitor()

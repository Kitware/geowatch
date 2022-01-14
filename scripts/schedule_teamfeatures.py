

def schedule_teamfeature_compute():
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension. Maybe airflow / luigi.
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

    commands = []
    commands.append(ub.codeblock(
        fr'''
        python -m watch.tasks.rutgers_material_seg.predict \
            --test_dataset="{base_coco_fpath}" \
            --checkpoint_fpath="{model_fpaths['rutgers_materials']}" \
            --pred_dataset="{outputs['rutgers_materials']}" \
            --default_config_key=iarpa \
            --num_workers="8" \
            --batch_size=32 --gpus "0" \
            --compress=DEFLATE --blocksize=64
        '''))

    commands.append(ub.codeblock(
        fr'''
        python -m watch.tasks.landcover.predict \
            --dataset="{base_coco_fpath}" \
            --deployed="{model_fpaths['dzyne_landcover']}" \
            --output="{outputs['dzyne_landcover']}" \
            --device=0 \
            --num_workers="8"
        '''))

    commands.append(ub.codeblock(
        fr'''
        python -m watch.tasks.invariants.predict \
            --input_kwcoco "{base_coco_fpath}" \
            --output_kwcoco "{outputs['uky_invariants']}" \
            --pretext_ckpt_path "{model_fpaths['uky_pretext']}" \
            --segmentation_ckpt "{model_fpaths['uky_segmentation']}" \
            --do_pca 1 \
            --num_dim 8 \
            --num_workers avail/2 \
            --write_workers avail/2 \
            --tasks all
        '''))


    tmux_queue.TMUXMultiQueue(size=2)

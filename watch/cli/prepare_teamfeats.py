"""
The following example simply produces the script under different variations.

CommandLine:
    xdoctest -m watch.cli.prepare_teamfeats __doc__

Example:
    >>> from watch.cli.prepare_teamfeats import *  # NOQA
    >>> dvc_dpath = ub.Path('.')
    >>> base_fpath = dvc_dpath / 'bundle/data.kwcoco.json'
    >>> config = {
    >>>     'base_fpath': './bundle/data.kwcoco.json',
    >>>     'gres': [0, 1],
    >>>     'dvc_dpath': './',
    >>> #
    >>>     'virtualenv_cmd': 'conda activate watch',
    >>> #
    >>>     'with_landcover': 1,
    >>>     'with_materials': 1,
    >>>     'with_invariants': 1,
    >>>     'do_splits': 1,
    >>> #
    >>>     'run': 0,
    >>>     #'check': False,
    >>>     'cache': False,
    >>>     'backend': 'serial',
    >>>     'verbose': 0,
    >>> }
    >>> config['backend'] = 'slurm'
    >>> queue = prep_feats(cmdline=False, **config)
    >>> queue.rprint(0, 0)
    >>> config['backend'] = 'tmux'
    >>> queue = prep_feats(cmdline=False, **config)
    >>> queue.rprint(0, 0)
    >>> config['backend'] = 'serial'
    >>> queue = prep_feats(cmdline=False, **config)
    >>> queue.rprint(0, 0)
"""


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

        'invariant_segmentation': scfg.Value(False, help='Enable/Disable segmentation part of invariants'),
        'invariant_pca': scfg.Value(0, help='Enable/Disable invariant PCA'),

        'virtualenv_cmd': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your bashrc
            does not start it by default.''')),

        'data_workers': scfg.Value(2, help='dataloader workers for each proc'),
        'depth_workers': scfg.Value(2, help='workers for depth only. On systems with < 32GB RAM might need to set to 0'),

        'keep_sessions': scfg.Value(False, help='if True does not close tmux sessions'),

        'workers': scfg.Value('auto', help='Maximum number of parallel jobs, 0 is no-nonsense serial mode. '),
        'run': scfg.Value(0, help='if True execute the pipeline'),
        'cache': scfg.Value(True, help='if True skip completed results'),

        'do_splits': scfg.Value(True, help='if True also make splits'),

        'follow': scfg.Value(True),

        'serial': scfg.Value(False, help='if True use serial mode'),

        'backend': scfg.Value('tmux', help=None),

        'check': scfg.Value(True, help='if True check files exist where we can'),
        'verbose': scfg.Value(1, help=''),
    }


def prep_feats(cmdline=True, **kwargs):
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.

    TODO:
        - [ ] Option to just dump the serial bash script that does everything.
    """
    from scriptconfig.smartcast import smartcast

    config = TeamFeaturePipelineConfig(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    gres = config['gres']
    # check = config['check']
    gres = smartcast(gres)
    if gres is None:
        gres = 'auto'
    print('gres = {!r}'.format(gres))
    if gres  == 'auto':
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

    if config['dvc_dpath'] == 'auto':
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
    else:
        dvc_dpath = ub.Path(config['dvc_dpath'])

    if config['base_fpath'] == 'auto':
        # Auto hack.
        # base_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
        base_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
    else:
        base_fpath = ub.Path(config['base_fpath'])

    aligned_bundle_dpath = base_fpath.parent

    if workers == 0:
        gres = None

    if gres is None:
        size = max(1, workers)
    else:
        size = len(gres)

    # queue = tmux_queue.TMUXMultiQueue(name='teamfeat', size=size, gres=gres)
    from watch.utils import cmd_queue
    queue = cmd_queue.Queue.create(
        name='watch-teamfeat',
        backend=config['backend'],
        # Tmux only
        size=size, gres=gres,
    )

    if config['virtualenv_cmd']:
        queue.add_header_command(config['virtualenv_cmd'])

    _populate_teamfeat_queue(queue, base_fpath, dvc_dpath,
                             aligned_bundle_dpath, config)

    if config['verbose']:
        queue.rprint()

    if config['run']:
        agg_state = None
        # follow = config['follow']
        # if follow and workers == 0 and len(queue.workers) == 1:
        #     queue = queue.workers[0]
        #     fpath = queue.write()
        #     ub.cmd(f'bash {fpath}', verbose=3, check=True)
        # else:
        if config['serial']:
            queue.serial_run()
        else:
            queue.run()
        if config['follow']:
            agg_state = queue.monitor()
        if not config['keep_sessions']:
            if agg_state is not None and not agg_state['errored']:
                queue.kill()

    """
    Ignore:
        python -m kwcoco stats data.kwcoco.json uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
        python -m watch stats uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
    """
    return queue


def _populate_teamfeat_queue(queue, base_fpath, dvc_dpath, aligned_bundle_dpath, config):
    from watch.utils.lightning_ext import util_globals
    data_workers = util_globals.coerce_num_workers(config['data_workers'])

    model_fpaths = {
        'rutgers_materials': dvc_dpath / 'models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth',
        # 'rutgers_materials': dvc_dpath / 'models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth',
        'dzyne_landcover': dvc_dpath / 'models/landcover/visnav_remap_s2_subset.pt',

        # 2022-02-11
        # 'uky_pretext': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pretext_package.pt',
        # 'uky_pca': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pca_projection_matrix.pt',
        # 'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_segmentation_model/segmentation_package.pt',

        # 2022-03-11
        # 'uky_pretext': dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_pretext_model/pretext_package.pt',
        # 'uky_pca': dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_pretext_model/pca_projection_matrix.pt',
        # 'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_segmentation_model/segmentation_package.pt',  # uses old segmentation model

        # 2022-03-21
        'uky_pretext': dvc_dpath / 'models/uky/uky_invariants_2022_03_21/pretext_model/pretext_package.pt',
        'uky_pca': dvc_dpath / 'models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt',
        # 'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_02_21/TA1_segmentation_model/segmentation_package.pt',  # uses old segmentation model

        # TODO: use v1 on RGB and v2 on PAN
        'dzyne_depth': dvc_dpath / 'models/depth/weights_v1.pt',
        # 'dzyne_depth': dvc_dpath / 'models/depth/weights_v2_gray.pt',
    }

    outputs = {
        'rutgers_materials': aligned_bundle_dpath / 'rutgers_material_seg_v3.kwcoco.json',
        'dzyne_landcover': aligned_bundle_dpath / 'dzyne_landcover.kwcoco.json',
        'dzyne_depth': aligned_bundle_dpath / 'dzyne_depth.kwcoco.json',
        'uky_invariants': aligned_bundle_dpath / 'uky_invariants.kwcoco.json',
    }

    print('Exist check: ')
    print('model_packages: ' + ub.repr2(ub.map_vals(lambda x: x.exists(), model_fpaths)))
    print('feature outputs: ' + ub.repr2(ub.map_vals(lambda x: x.exists(), outputs)))

    # TODO: different versions of features need different codes.
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
        task['gpus'] = 1
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.landcover.predict \
                --dataset="{base_fpath}" \
                --deployed="{model_fpaths['dzyne_landcover']}" \
                --output="{task['output_fpath']}" \
                --num_workers="{data_workers}" \
                --select_images='.sensor_coarse == "S2"' \
                --device=0
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    key = 'with_depth'
    if config[key]:
        # Landcover is fairly fast to run, do it first
        task = {}
        # Only need 1 worker to minimize lag between images, task is GPU bound
        depth_data_workers = config['depth_workers']
        if depth_data_workers == 'auto':
            import psutil
            import pint
            reg = pint.UnitRegistry()
            vmem_info = psutil.virtual_memory()
            total_gb = (vmem_info.total * reg.byte).to(reg.gigabyte).m
            avail_gb = (vmem_info.available * reg.byte).to(reg.gigabyte).m
            if avail_gb < 32:
                depth_data_workers = 0
            elif avail_gb < 64:
                depth_data_workers = 1
            else:
                depth_data_workers = 2
            print('total_gb = {!r}'.format(total_gb))
            print('avail_gb = {!r}'.format(avail_gb))

        # depth_data_workers = min(2, data_workers)
        depth_window_size = 512  # takes 18GB
        task['output_fpath'] = outputs['dzyne_depth']
        task['gpus'] = 1
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.depth.predict \
                --dataset="{base_fpath}" \
                --output="{task['output_fpath']}" \
                --deployed="{model_fpaths['dzyne_depth']}" \
                --data_workers={depth_data_workers} \
                --window_size={depth_window_size} \
                --cache=1
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    # Run materials while landcover is running
    key = 'with_materials'
    if config[key]:
        task = {}
        task['output_fpath'] = outputs['rutgers_materials']
        task['gpus'] = 1
        # --export_raw_features=1 \
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.rutgers_material_seg.predict \
                --test_dataset="{base_fpath}" \
                --checkpoint_fpath="{model_fpaths['rutgers_materials']}" \
                --pred_dataset="{task['output_fpath']}" \
                --default_config_key=iarpa \
                --num_workers="{data_workers}" \
                --batch_size=32 --gpus "0" \
                --compress=DEFLATE --blocksize=128 --cache=True
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    # When landcover finishes run invariants
    # Note: Does not run on a 1080, needs 18GB in this form
    key = 'with_invariants'
    if config[key]:
        task = {}

        if config['invariant_segmentation']:
            # segmentation_parts = [
            #     rf'''
            #     --segmentation_package_path "{model_fpaths['uky_segmentation']}"
            #     '''
            # ]
            raise NotImplementedError()

        if not model_fpaths['uky_pretext'].exists():
            print('Warning: UKY pretext model does not exist')

        # all_tasks = 'before_after segmentation pretext'
        task['output_fpath'] = outputs['uky_invariants']
        task['gpus'] = 1
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.invariants.predict \
                --input_kwcoco "{base_fpath}" \
                --output_kwcoco "{task['output_fpath']}" \
                --pretext_package_path "{model_fpaths['uky_pretext']}" \
                --pca_projection_path  "{model_fpaths['uky_pca']}" \
                --do_pca {config['invariant_pca']} \
                --patch_overlap=0.5 \
                --num_workers="{data_workers}" \
                --write_workers 2 \
                --tasks before_after pretext
            ''')
        combo_code_parts.append(codes[key])
        tasks.append(task)

    task_jobs = []
    for task in tasks:
        if config['cache']:
            if not task['output_fpath'].exists():
                # command = f"[[ -f '{task['output_fpath']}' ]] || " + task['command']
                command = f"test -f '{task['output_fpath']}' || " + task['command']
                job = queue.submit(command, gpus=task['gpus'])
                task_jobs.append(job)
        else:
            job = queue.submit(task['command'])
            task_jobs.append(job)

    # Finalize features by combining them all into combo.kwcoco.json
    tocombine = [str(base_fpath)] + [str(task['output_fpath']) for task in tasks]
    combo_code = ''.join(sorted(combo_code_parts))

    base_combo_fpath = aligned_bundle_dpath / f'combo_{combo_code}.kwcoco.json'

    # Note: sync tells the queue that everything after this
    # depends on everything before this
    queue.sync()

    src_lines = ' \\\n        '.join(tocombine)
    command = '\n'.join([
        'python -m watch.cli.coco_combine_features \\',
        f'    --src {src_lines} \\',
        f'    --dst {base_combo_fpath}'
    ])
    print('task_jobs = {!r}'.format(task_jobs))
    queue.submit(command)

    if config['do_splits']:
        # Also call the prepare-splits script
        from watch.cli import prepare_splits
        base_fpath = str(base_combo_fpath)
        queue.sync()
        prepare_splits._submit_split_jobs(base_fpath, queue)

    return queue

main = prep_feats

if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
        python -m watch.cli.prepare_teamfeats \
            --base_fpath="$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json" \
            --gres=0 \
            --with_depth=0 \
            --run=False --cache=False --virtualenv_cmd "conda activate watch" \
            --backend=serial

        python -m watch.cli.prepare_teamfeats --gres=0,2 --with_depth=True --keep_sessions=True
        python -m watch.cli.prepare_teamfeats --gres=2 --with_materials=False --keep_sessions=True

        # TODO: rename to schedule teamfeatures

        # TO UPDATE ANNOTS
        # Update to whatever the state of the annotations submodule is
        DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
        python -m watch project_annotations \
            --src $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --site_models="$DVC_DPATH/annotations/site_models/*.geojson"

        kwcoco stats $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data_20220203.kwcoco.json $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json

        # Team Features on Drop2
        DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --gres=0,1 --with_depth=0 --with_materials=False  --with_invariants=False \
            --run=0 --do_splits=True

        ###
        DATASET_CODE=Aligned-Drop2-TA1-2022-02-24
        DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
        DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --gres=0,1 \
            --with_depth=1 \
            --with_landcover=1 \
            --with_invariants=1 \
            --with_materials=1 \
            --depth_workers=auto \
            --do_splits=1  --cache=0 --run=0

        ###
        DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
        DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --gres=0,1 \
            --with_depth=0 \
            --with_landcover=1 \
            --with_invariants=1 \
            --with_materials=1 \
            --depth_workers=auto \
            --invariant_pca=0 \
            --invariant_segmentation=0 \
            --do_splits=0  --cache=1 --run=0

        # Simple demo
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=./mydata/data.kwcoco.json \
            --gres=0,1 \
            --do_splits=0 \
            --with_depth=0 \
            --with_landcover=1 \
            --with_invariants=0 \
            --with_materials=1 \
            --cache=0 \
            --backend=serial \
            --run=0


    """
    main(cmdline=True)

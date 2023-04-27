#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
The following example simply produces the script under different variations.

CommandLine:
    xdoctest -m watch.cli.prepare_teamfeats __doc__

SeeAlso:
    ../tasks/invariants/predict.py
    ../tasks/landcover/predict.py
    ../tasks/depth/predict.py
    ../tasks/cold/predict.py

    ~/code/watch/dev/poc/prepare_time_combined_dataset.py

Example:
    >>> from watch.cli.prepare_teamfeats import *  # NOQA
    >>> expt_dvc_dpath = ub.Path('./pretend_expt_dpath')
    >>> config = {
    >>>     'base_fpath': './pretend_bundle/data.kwcoco.json',
    >>>     'gres': [0, 1],
    >>>     'expt_dvc_dpath': './pretend_expt_dvc',
    >>> #
    >>>     'virtualenv_cmd': 'conda activate watch',
    >>> #
    >>>     'with_landcover': 1,
    >>>     'with_materials': 1,
    >>>     'with_invariants': 1,
    >>> #
    >>>     'run': 0,
    >>>     #'check': False,
    >>>     'skip_existing': False,
    >>>     'backend': 'serial',
    >>> }
    >>> config['backend'] = 'slurm'
    >>> queue = prep_feats(cmdline=False, **config)
    >>> queue.print_commands(0, 0)
    >>> config['backend'] = 'tmux'
    >>> queue = prep_feats(cmdline=False, **config)
    >>> queue.print_commands(0, 0)
    >>> config['backend'] = 'serial'
    >>> queue = prep_feats(cmdline=False, **config)
    >>> queue.print_commands(0, 0)

Ignore:

    # Drop 6
    export CUDA_VISIBLE_DEVICES="0,1"
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    python -m watch.cli.prepare_teamfeats \
        --base_fpath "$BUNDLE_DPATH"/imganns-*.kwcoco.zip \
        --expt_dvc_dpath="$DVC_EXPT_DPATH" \
        --with_invariants2=0 \
        --with_landcover=0 \
        --with_materials=0 \
        --with_invariants=0 \
        --with_depth=0 \
        --with_cold=1 \
        --skip_existing=1 \
        --gres=0,1 --tmux_workers=4 --backend=tmux --run=0 --print-commands

    # Drop 6
    export CUDA_VISIBLE_DEVICES="0,1"
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    python -m watch.cli.prepare_teamfeats \
        --base_fpath "$BUNDLE_DPATH"/imganns-KR_R00*.kwcoco.zip \
        --expt_dvc_dpath="$DVC_EXPT_DPATH" \
        --with_invariants2=1 \
        --with_landcover=0 \
        --with_materials=0 \
        --with_invariants=0 \
        --with_depth=0 \
        --with_cold=0 \
        --skip_existing=1 \
        --assets_dname=teamfeats \
        --gres=0,1 --tmux_workers=4 --backend=tmux --run=0
"""
import scriptconfig as scfg
import ubelt as ub
from cmd_queue.cli_boilerplate import CMDQueueConfig


class TeamFeaturePipelineConfig(CMDQueueConfig):
    """
    This generates the bash commands necessary to run team feature computation,
    followed by aggregation and then splitting out train / val datasets.

    Note:
        The models and parameters to use are hard coded in this script.

    TODO:
        - [ ] jsonargparse use-case: specifying parmeters of the subalgos
    """
    base_fpath = scfg.Value(None, help=ub.paragraph(
            '''
            One or more base coco files to compute team-features on.
            '''), nargs='+', group='inputs')
    expt_dvc_dpath = scfg.Value('auto', help=ub.paragraph(
            '''
            The DVC directory where team feature model weights can be
            found. If "auto" uses the
            ``watch.find_dvc_dpath(tags='phase2_expt')`` mechanism to
            infer the location.
            '''), nargs=None, group='inputs')

    gres = scfg.Value('auto', help='comma separated list of gpus or auto', nargs=None, group='cmd-queue')

    with_landcover = scfg.Value(False, help='Include DZYNE landcover features', nargs=None, group='team feature enablers')
    with_materials = scfg.Value(False, help='Include Rutgers material features', nargs=None, group='team feature enablers')
    with_invariants = scfg.Value(False, help='Include UKY invariant features', nargs=None, group='team feature enablers')
    with_invariants2 = scfg.Value(False, help='Include UKY invariant features', nargs=None, group='team feature enablers')
    with_depth = scfg.Value(False, help='Include DZYNE WorldView depth features', nargs=None, group='team feature enablers')
    with_cold = scfg.Value(False, help='Include COLD features', nargs=None)

    invariant_segmentation = scfg.Value(False, help=ub.paragraph(
            '''
            Enable/Disable segmentation part of invariants
            '''), nargs=None, group='invariants options')
    invariant_pca = scfg.Value(0, help='Enable/Disable invariant PCA', nargs=None, group='invariants options')
    invariant_resolution = scfg.Value('10GSD', help='GSD for invariants', nargs=None, group='invariants options')

    virtualenv_cmd = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your
            bashrc does not start it by default.
            '''), nargs=None)

    skip_existing = scfg.Value(True, help='if True skip completed results', nargs=None, group='common options')

    data_workers = scfg.Value(2, help='dataloader workers for each proc', nargs=None, group='common options')

    kwcoco_ext = scfg.Value('.kwcoco.zip', help=ub.paragraph(
            '''
            use .kwcoco.json or .kwcoco.zip for outputs
            '''), nargs=None, group='common options')

    assets_dname = scfg.Value('_teamfeats', help=ub.paragraph(
        '''
        The name of the top-level directory to write new assets.
        '''), group='common options')

    check = scfg.Value(True, help='if True check files exist where we can', nargs=None, group='common options')

    cold_workers = scfg.Value(4, help='workers for pycold', nargs=None, group='cold options')
    cold_workermode = scfg.Value('process', help='workers mode for pycold', nargs=None, group='cold options')

    depth_workers = scfg.Value(2, help=ub.paragraph(
            '''
            workers for depth only. On systems with < 32GB RAM might
            need to set to 0
            '''), nargs=None, group='depth options')

    # keep_sessions = scfg.Value(False, help='if True does not close tmux sessions', nargs=None)

    # workers = scfg.Value('auto', help=ub.paragraph(
    #         '''
    #         Maximum number of parallel jobs, 0 is no-nonsense serial
    #         mode.
    #         '''), nargs=None)
    # run = scfg.Value(0, help='if True execute the pipeline', nargs=None)
    # serial = scfg.Value(False, help='if True use serial mode', nargs=None)
    # backend = scfg.Value('tmux', help=None, nargs=None)


def prep_feats(cmdline=True, **kwargs):
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.

    TODO:
        - [ ] Option to just dump the serial bash script that does everything.
    """
    config = TeamFeaturePipelineConfig.cli(cmdline=cmdline, data=kwargs,
                                           strict=True)
    import rich
    rich.print('config = {}'.format(ub.urepr(config, nl=2)))
    from scriptconfig.smartcast import smartcast
    from watch.utils import util_path

    # hack for cmd-queue, will be fixed soon
    config.slurm_options = config.slurm_options or {}

    gres = smartcast(config['gres'])
    if gres is None:
        gres = 'auto'
    if gres == 'auto':
        import torch
        gres = list(range(torch.cuda.device_count()))
    elif not ub.iterable(gres):
        gres = [gres]

    if config['expt_dvc_dpath'] == 'auto':
        import watch
        expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    else:
        expt_dvc_dpath = ub.Path(config['expt_dvc_dpath'])

    from watch.mlops.old import pipeline_v1
    pipeline = pipeline_v1.Pipeline()

    blocklist = [
        '_dzyne_landcover',
        '_uky_invariants',
    ]

    base_fpath_pat = config['base_fpath']
    for base_fpath in util_path.coerce_patterned_paths(base_fpath_pat):

        # Hack to prevent doubling up.
        # Should really just choose a better naming scheme so we don't have
        # to break user expectations about glob
        if any(b in base_fpath.name for b in blocklist):
            print(f'blocked base_fpath={base_fpath}')
            continue

        if config['check']:
            if not base_fpath.exists():
                raise FileNotFoundError(
                    'Specified kwcoco file: {base_fpath!r=} does not exist and check=True')
        aligned_bundle_dpath = base_fpath.parent

        _populate_teamfeat_queue(pipeline, base_fpath, expt_dvc_dpath,
                                 aligned_bundle_dpath, config)

    queue = config.create_queue(gres=gres)

    pipeline._populate_explicit_dependency_queue(queue)
    # pipeline._populate_implicit_dependency_queue(queue, skip_existing=config['skip_existing'])

    config.run_queue(queue)

    """
    Ignore:
        python -m kwcoco stats data.kwcoco.json uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
        python -m watch stats uky_invariants.kwcoco.json dzyne_landcover.kwcoco.json
    """
    return queue


def _populate_teamfeat_queue(pipeline, base_fpath, expt_dvc_dpath, aligned_bundle_dpath, config):
    from watch.utils import util_parallel
    from watch.utils import simple_dvc
    data_workers = util_parallel.coerce_num_workers(config['data_workers'])

    model_fpaths = {
        'rutgers_materials': expt_dvc_dpath / 'models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth',
        # 'rutgers_materials': dvc_dpath / 'models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth',
        # 'dzyne_landcover': expt_dvc_dpath / 'models/landcover/visnav_remap_s2_subset.pt',
        'dzyne_landcover': expt_dvc_dpath / 'models/landcover/sentinel2.pt',

        # 2022-02-11
        # 'uky_pretext': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pretext_package.pt',
        # 'uky_pca': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_pretext_model/pca_projection_matrix.pt',
        # 'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_segmentation_model/segmentation_package.pt',

        # 2022-03-11
        # 'uky_pretext': dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_pretext_model/pretext_package.pt',
        # 'uky_pca': dvc_dpath / 'models/uky/uky_invariants_2022_03_11/TA1_pretext_model/pca_projection_matrix.pt',
        # 'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_02_11/TA1_segmentation_model/segmentation_package.pt',  # uses old segmentation model

        # 2022-03-21
        'uky_pretext': expt_dvc_dpath / 'models/uky/uky_invariants_2022_03_21/pretext_model/pretext_package.pt',
        'uky_pca': expt_dvc_dpath / 'models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt',

        'uky_pretext2': expt_dvc_dpath / 'models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt',
        # 'uky_segmentation': dvc_dpath / 'models/uky/uky_invariants_2022_02_21/TA1_segmentation_model/segmentation_package.pt',  # uses old segmentation model

        # TODO: use v1 on RGB and v2 on PAN
        'dzyne_depth': expt_dvc_dpath / 'models/depth/weights_v1.pt',
        # 'dzyne_depth': dvc_dpath / 'models/depth/weights_v2_gray.pt',
    }

    subset_name = base_fpath.name.split('.')[0]

    name_suffix = '_' + ub.hash_data(base_fpath)[0:8]

    outputs = {
        'rutgers_materials': aligned_bundle_dpath / (subset_name + '_rutgers_material_seg_v3' + config['kwcoco_ext']),
        'dzyne_landcover': aligned_bundle_dpath / (subset_name + '_dzyne_landcover' + config['kwcoco_ext']),
        'dzyne_depth': aligned_bundle_dpath / (subset_name + '_dzyne_depth' + config['kwcoco_ext']),
        'uky_invariants': aligned_bundle_dpath / (subset_name + '_uky_invariants' + config['kwcoco_ext']),
        'cold': aligned_bundle_dpath / (subset_name + '_cold' + config['kwcoco_ext']),
    }

    # print('Exist check: ')
    # print('model_packages: ' + ub.urepr(ub.map_vals(lambda x: x.exists(), model_fpaths)))
    # print('feature outputs: ' + ub.urepr(ub.map_vals(lambda x: x.exists(), outputs)))

    # TODO: different versions of features need different codes.
    codes = {
        'with_landcover': 'L',
        'with_depth': 'D',
        'with_materials': 'M',
        'with_invariants': 'I',
        'with_invariants2': 'I2',
        'with_cold': 'C',
    }

    # tmux queue is still limited. The order of submission matters.
    task_jobs = []

    combo_code_parts = []
    key = 'with_landcover'
    if config[key]:
        simple_dvc.SimpleDVC().request(model_fpaths['dzyne_landcover'])

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
                --with_hidden=32 \
                --select_images='.sensor_coarse == "S2"' \
                --assets_dname="{config.assets_dname}" \
                --device=0
            ''')
        combo_code_parts.append(codes[key])
        job = pipeline.submit(
            name='landcover' + name_suffix,
            command=task['command'],
            in_paths=[base_fpath],
            out_paths={
                'output_fpath': task['output_fpath']
            },
        )
        task_jobs.append(job)

    key = 'with_cold'
    if config[key]:
        # Landcover is fairly fast to run, do it first
        task = {}
        task['output_fpath'] = outputs['cold']
        task['gpus'] = 0
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.cold.predict \
                --coco_fpath="{base_fpath}" \
                --mod_coco_fpath="{task['output_fpath']}" \
                --sensors='L8' \
                --adj_cloud=False \
                --method='COLD' \
                --prob=0.99 \
                --conse=6 \
                --cm_interval=60 \
                --year_lowbound=None \
                --year_highbound=None \
                --coefs=cv,rmse,a0,a1,b1,c1 \
                --coefs_bands=0,1,2,3,4,5 \
                --timestamp=False \
                --combine=False \
                --resolution=10GSD \
                --workermode="{config.cold_workermode}" \
                --workers="{config.cold_workers}"
            ''')
        # --coefs=cv,a0,a1,b1,c1,rmse \
        combo_code_parts.append(codes[key])
        job = pipeline.submit(
            name='cold' + name_suffix,
            command=task['command'],
            in_paths=[base_fpath],
            out_paths={
                'output_fpath': task['output_fpath']
            },
        )
        task_jobs.append(job)

    key = 'with_depth'
    if config[key]:
        simple_dvc.SimpleDVC().request(model_fpaths['dzyne_depth'])

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
        depth_window_size = 1440
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
                --skip_existing=1
            ''')
        combo_code_parts.append(codes[key])
        job = pipeline.submit(
            name='depth' + name_suffix,
            command=task['command'],
            in_paths=[base_fpath],
            out_paths={
                'output_fpath': task['output_fpath']
            },
        )
        task_jobs.append(job)

    # Run materials while landcover is running
    key = 'with_materials'
    if config[key]:
        simple_dvc.SimpleDVC().request(model_fpaths['rutgers_materials'])

        task = {}
        task['output_fpath'] = outputs['rutgers_materials']
        task['gpus'] = 1
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.rutgers_material_seg.predict \
                --test_dataset="{base_fpath}" \
                --checkpoint_fpath="{model_fpaths['rutgers_materials']}" \
                --pred_dataset="{task['output_fpath']}" \
                --default_config_key=iarpa \
                --num_workers="{data_workers}" \
                --export_raw_features=1 \
                --batch_size=32 --gpus "0" \
                --compress=DEFLATE --blocksize=128 --skip_existing=True
            ''')
        combo_code_parts.append(codes[key])
        job = pipeline.submit(
            name='materials' + name_suffix,
            command=task['command'],
            in_paths=[base_fpath],
            out_paths={
                'output_fpath': task['output_fpath']
            },
        )
        task_jobs.append(job)

    # When landcover finishes run invariants
    # Note: Does not run on a 1080, needs 18GB in this form
    key = 'with_invariants'
    if config[key]:
        raise Exception('Use with_invariants2 instead')
        task = {}
        simple_dvc.SimpleDVC().request(model_fpaths['uky_pretext'])

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
                --input_space_scale=10GSD \
                --window_space_scale=10GSD \
                --patch_size=256 \
                --do_pca {config['invariant_pca']} \
                --patch_overlap=0.3 \
                --workers="{data_workers}" \
                --io_workers 2 \
                --tasks before_after pretext
            ''')
        combo_code_parts.append(codes[key])
        job = pipeline.submit(
            name='invariants' + name_suffix,
            command=task['command'],
            in_paths=[base_fpath],
            out_paths={
                'output_fpath': task['output_fpath']
            },
        )
        task_jobs.append(job)

    key = 'with_invariants2'
    if config[key]:
        simple_dvc.SimpleDVC().request(model_fpaths['uky_pretext2'])
        task = {}
        if not model_fpaths['uky_pretext2'].exists():
            print('Warning: UKY pretext model does not exist')
        # all_tasks = 'before_after segmentation pretext'
        task['output_fpath'] = outputs['uky_invariants']
        task['gpus'] = 1
        # --input_kwcoco=$DVC_DATA_DPATH/Drop4-BAS/data_train.kwcoco.json \
        # --output_kwcoco=$DVC_DATA_DPATH/Drop4-BAS/data_train_invar13.kwcoco.json \
        # --pretext_package=$DVC_EXPT_DPATH/models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt \
        task['command'] = ub.codeblock(
            fr'''
            python -m watch.tasks.invariants.predict \
                --input_kwcoco "{base_fpath}" \
                --output_kwcoco "{task['output_fpath']}" \
                --pretext_package_path "{model_fpaths['uky_pretext2']}" \
                --pca_projection_path  "{model_fpaths['uky_pca']}" \
                --input_resolution={config['invariant_resolution']} \
                --window_resolution={config['invariant_resolution']} \
                --patch_size=256 \
                --do_pca {config['invariant_pca']} \
                --patch_overlap=0.3 \
                --workers="{data_workers}" \
                --io_workers 0 \
                --assets_dname="{config.assets_dname}" \
                --tasks before_after pretext
            ''')
        combo_code_parts.append(codes[key])
        job = pipeline.submit(
            name='invariants2' + name_suffix,
            command=task['command'],
            in_paths=[base_fpath],
            out_paths={
                'output_fpath': task['output_fpath']
            },
        )
        task_jobs.append(job)

    # for task in tasks:
    #     # if config['skip_existing']:
    #     #     if not task['output_fpath'].exists():
    #     #         # command = f"[[ -f '{task['output_fpath']}' ]] || " + task['command']
    #     #         command = f"test -f '{task['output_fpath']}' || " + task['command']
    #     #         job = queue.submit(command, gpus=task['gpus'])
    #     #         task_jobs.append(job)
    #     # else:
    #     #     job = queue.submit(task['command'])
    #     #     task_jobs.append(job)

    # Finalize features by combining them all into combo.kwcoco.json
    # tocombine = [str(base_fpath)] + [str(task['output_fpath']) for task in tasks]

    feature_paths = [str(job.out_paths['output_fpath']) for job in task_jobs]
    tocombine = [str(base_fpath)] + feature_paths
    combo_code = ''.join(sorted(combo_code_parts))

    base_combo_fpath = aligned_bundle_dpath / (f'combo_{subset_name}_{combo_code}' + config['kwcoco_ext'])

    # Note: sync tells the queue that everything after this
    # depends on everything before this
    # queue.sync()

    if config.skip_existing:
        request_jobs = []
        for job in task_jobs:
            if not all(p.exists() for p in job.out_paths.values()):
                request_jobs.append(job)
    else:
        request_jobs = task_jobs

    src_lines = ' \\\n        '.join(tocombine)
    command = '\n'.join([
        'python -m watch.cli.coco_combine_features \\',
        f'    --src {src_lines} \\',
        f'    --dst {base_combo_fpath}'
    ])
    pipeline.submit(
        name='combine_features' + name_suffix,
        command=command,
        in_paths=feature_paths,
        out_paths={
            'combo_fpath': base_combo_fpath,
        },
        depends=request_jobs
    )


main = prep_feats

if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(smartwatch_dvc)
        python -m watch.cli.prepare_teamfeats \
            --base_fpath="$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json" \
            --gres=0 \
            --with_depth=0 \
            --run=False --skip_existing=False --virtualenv_cmd "conda activate watch" \
            --backend=serial

        python -m watch.cli.prepare_teamfeats --gres=0,2 --with_depth=True --keep_sessions=True
        python -m watch.cli.prepare_teamfeats --gres=2 --with_materials=False --keep_sessions=True

        # TODO: rename to schedule teamfeatures

        # TO UPDATE ANNOTS
        # Update to whatever the state of the annotations submodule is
        DVC_DPATH=$(smartwatch_dvc)
        python -m watch reproject_annotations \
            --src $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --site_models="$DVC_DPATH/annotations/site_models/*.geojson"

        kwcoco stats $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data_20220203.kwcoco.json $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json

        # Team Features on Drop2
        DVC_DPATH=$(smartwatch_dvc)
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --gres=0,1 --with_depth=0 --with_materials=False  --with_invariants=False \
            --run=0

        ###
        DATASET_CODE=Aligned-Drop2-TA1-2022-02-24
        DVC_DPATH=$(smartwatch_dvc)
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
            --skip_existing=0 --run=0

        ###
        DVC_DPATH=$(smartwatch_dvc)
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
            --skip_existing=1 --run=0

        # Simple demo
        python -m watch.cli.prepare_teamfeats \
            --base_fpath=./mydata/data.kwcoco.json \
            --gres=0,1 \
            --with_depth=0 \
            --with_landcover=1 \
            --with_invariants=0 \
            --with_materials=1 \
            --skip_existing=0 \
            --backend=tmux \
            --run=0


    """
    main(cmdline=True)

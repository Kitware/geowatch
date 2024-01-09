#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
The following example simply produces the script under different variations.

CommandLine:
    xdoctest -m geowatch.cli.prepare_teamfeats __doc__

SeeAlso:
    ../tasks/invariants/predict.py
    ../tasks/landcover/predict.py
    ../tasks/depth/predict.py
    ../tasks/cold/predict.py

    ~/code/watch/dev/poc/prepare_time_combined_dataset.py

Example:
    >>> from geowatch.cli.prepare_teamfeats import *  # NOQA
    >>> expt_dvc_dpath = ub.Path('./pretend_expt_dpath')
    >>> config = {
    >>>     'src_kwcocos': './pretend_bundle/data.kwcoco.json',
    >>>     'gres': [0, 1],
    >>>     'expt_dvc_dpath': './pretend_expt_dvc',
    >>> #
    >>>     'virtualenv_cmd': 'conda activate geowatch',
    >>> #
    >>>     #'with_s2_landcover': 1,
    >>>     #'with_materials': 1,
    >>>     #'with_invariants2': 1,
    >>>     'with_mae': 1,
    >>> #
    >>>     'run': 0,
    >>>     'check': False,
    >>>     'skip_existing': False,
    >>>     'backend': 'serial',
    >>> }
    >>> config['backend'] = 'slurm'
    >>> outputs = prep_feats(cmdline=False, **config)
    >>> outputs['queue'].print_commands(0, 0)
    >>> config['backend'] = 'tmux'
    >>> outputs = prep_feats(cmdline=False, **config)
    >>> outputs['queue'].print_commands(0, 0)
    >>> config['backend'] = 'serial'
    >>> outputs = prep_feats(cmdline=False, **config)
    >>> outputs['queue'].print_commands(0, 0)


Example:
    >>> # Test landcover commands
    >>> from geowatch.cli.prepare_teamfeats import *  # NOQA
    >>> expt_dvc_dpath = ub.Path('./pretend_expt_dpath')
    >>> config = {
    >>>     'src_kwcocos': './PRETEND_BUNDLE/data.kwcoco.json',
    >>>     'gres': [0, 1],
    >>>     'expt_dvc_dpath': './PRETEND_EXPT_DVC',
    >>>     'virtualenv_cmd': 'conda activate geowatch',
    >>>     'with_s2_landcover': 1,
    >>>     'with_wv_landcover': 1,
    >>>     'num_wv_landcover_hidden': 0,
    >>>     'num_s2_landcover_hidden': 0,
    >>>     'run': 0,
    >>>     'check': False,
    >>>     'skip_existing': False,
    >>>     'backend': 'serial',
    >>> }
    >>> config['backend'] = 'serial'
    >>> outputs = prep_feats(cmdline=False, **config)
    >>> outputs['queue'].print_commands(0, 0)
    >>> output_paths = outputs['final_output_paths']
    >>> print('output_paths = {}'.format(ub.urepr(output_paths, nl=1)))

Ignore:

    # Drop 6
    export CUDA_VISIBLE_DEVICES="0,1"
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    python -m geowatch.cli.prepare_teamfeats \
        --src_kwcocos "$BUNDLE_DPATH"/imganns-*.kwcoco.zip \
        --expt_dvc_dpath="$DVC_EXPT_DPATH" \
        --with_invariants2=0 \
        --with_s2_landcover=0 \
        --with_materials=0 \
        --with_depth=0 \
        --with_cold=1 \
        --skip_existing=1 \
        --gres=0,1 --tmux_workers=4 --backend=tmux --run=0 --print-commands

    # Drop 6
    export CUDA_VISIBLE_DEVICES="0,1"
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    python -m geowatch.cli.prepare_teamfeats \
        --src_kwcocos "$BUNDLE_DPATH"/imganns-KR_R00*.kwcoco.zip \
        --expt_dvc_dpath="$DVC_EXPT_DPATH" \
        --with_invariants2=1 \
        --with_s2_landcover=0 \
        --with_materials=0 \
        --with_depth=0 \
        --with_cold=0 \
        --skip_existing=1 \
        --assets_dname=teamfeats \
        --gres=0,1 --tmux_workers=4 --backend=tmux --run=0


    # TimeCombined V2
    export CUDA_VISIBLE_DEVICES="0,1"
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
    BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2
    python -m geowatch.cli.prepare_teamfeats \
        --src_kwcocos "$BUNDLE_DPATH"/imganns-*[0-9].kwcoco.zip \
        --expt_dvc_dpath="$DVC_EXPT_DPATH" \
        --with_s2_landcover=1 \
        --with_invariants2=1 \
        --with_sam=1 \
        --with_materials=0 \
        --with_depth=0 \
        --with_cold=0 \
        --skip_existing=1 \
        --assets_dname=teamfeats \
        --gres=0, --tmux_workers=1 --backend=tmux --run=0

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    python -m geowatch.cli.prepare_splits \
        --src_kwcocos=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns*_I2LS*.kwcoco.zip \
        --constructive_mode=True \
        --suffix=I2LS \
        --backend=tmux --tmux_workers=6 \
        --run=1

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
    TRUE_SITE_DPATH=$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models
    OUTPUT_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2
    python -m geowatch reproject \
        --src $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/data_vali_I2LS_split6.kwcoco.zip \
        --inplace \
        --site_models=$TRUE_SITE_DPATH

    python -m geowatch reproject \
        --src $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/data_train_I2LS_split6.kwcoco.zip \
        --inplace \
        --site_models=$TRUE_SITE_DPATH
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
    src_kwcocos = scfg.Value(None, help=ub.paragraph(
            '''
            One or more base coco files to compute team-features on.
            '''), nargs='+', alias=['base_fpath'], group='inputs')
    expt_dvc_dpath = scfg.Value('auto', help=ub.paragraph(
            '''
            The DVC directory where team feature model weights can be
            found. If "auto" uses the
            ``geowatch.find_dvc_dpath(tags='phase2_expt')`` mechanism to
            infer the location.
            '''), group='inputs')

    gres = scfg.Value('auto', help='comma separated list of gpus or auto', group='cmd-queue')

    with_s2_landcover = scfg.Value(False, help='Include DZYNE S2 landcover features', group='team feature enablers')
    with_wv_landcover = scfg.Value(False, help='Include DZYNE WV landcover features', group='team feature enablers')
    with_materials = scfg.Value(False, help='Include Rutgers material features', group='team feature enablers')
    with_mae = scfg.Value(False, help='Include WU MAE features', group='team feature enablers')
    with_invariants2 = scfg.Value(False, help='Include UKY invariant features', group='team feature enablers')
    with_depth = scfg.Value(False, help='Include DZYNE WorldView depth features', group='team feature enablers')
    with_cold = scfg.Value(False, help='Include COLD features')
    with_sam = scfg.Value(False, help='Include SAM features')

    num_s2_landcover_hidden = 32
    num_wv_landcover_hidden = 32

    invariant_segmentation = scfg.Value(False, help=ub.paragraph(
            '''
            Enable/Disable segmentation part of invariants
            '''), group='invariants options')
    invariant_pca = scfg.Value(0, help='Enable/Disable invariant PCA', group='invariants options')
    invariant_resolution = scfg.Value('10GSD', help='GSD for invariants', group='invariants options')

    virtualenv_cmd = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your
            bashrc does not start it by default.
            '''))

    skip_existing = scfg.Value(True, help='if True skip completed results', group='common options')

    data_workers = scfg.Value(2, help='dataloader workers for each proc', group='common options')

    kwcoco_ext = scfg.Value('.kwcoco.zip', help=ub.paragraph(
            '''
            use .kwcoco.json or .kwcoco.zip for outputs
            '''), group='common options')

    assets_dname = scfg.Value('_teamfeats', help=ub.paragraph(
        '''
        The name of the top-level directory to write new assets.
        '''), group='common options')

    check = scfg.Value(True, help='if True check files exist where we can', group='common options')

    cold_workers = scfg.Value(4, help='workers for pycold', group='cold options')
    cold_workermode = scfg.Value('process', help='workers mode for pycold', group='cold options')

    depth_workers = scfg.Value(2, help=ub.paragraph(
            '''
            workers for depth only. On systems with < 32GB RAM might
            need to set to 0
            '''), group='depth options')


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
    from kwutil import util_path

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
        import geowatch
        expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    else:
        expt_dvc_dpath = ub.Path(config['expt_dvc_dpath'])

    blocklist = [
        '_dzyne_landcover',
        '_dzyne_s2_landcover',
        '_dzyne_wv_landcover',
        '_uky_invariants',
        '_rutgers_material_seg_v4',
    ]

    base_fpath_pat = config['src_kwcocos']
    base_fpath_list = list(util_path.coerce_patterned_paths(
        base_fpath_pat, globfallback=True))

    from geowatch.mlops.pipeline_nodes import Pipeline

    dag_nodes = []
    final_output_paths = []

    for src_fpath in base_fpath_list:
        # Hack to prevent doubling up.
        # Should really just choose a better naming scheme so we don't have
        # to break user expectations about glob
        if any(b in src_fpath.name for b in blocklist):
            print(f'blocked src_fpath={src_fpath}')
            continue

        if config.check:
            if not src_fpath.exists():
                raise FileNotFoundError(
                    'Specified kwcoco file: {src_fpath!r=} does not exist and check=True')
        aligned_bundle_dpath = src_fpath.parent

        nodes, base_combo_fpath = _make_teamfeat_nodes(
            src_fpath, expt_dvc_dpath,
            aligned_bundle_dpath, config)
        final_output_paths.append(base_combo_fpath)
        dag_nodes.extend(nodes)

    dag = Pipeline(dag_nodes)
    dag.configure(cache=True)

    queue = config.create_queue(gres=gres)
    dag.submit_jobs(
        queue=queue,
        skip_existing=config['skip_existing'],
        enable_links=False,
        write_invocations=False,
        write_configs=False,
    )

    # pipeline._populate_explicit_dependency_queue(queue)
    config.run_queue(queue)

    outputs = {
        'queue': queue,
        'final_output_paths': final_output_paths,
    }
    return outputs


def _make_teamfeat_nodes(src_fpath, expt_dvc_dpath, aligned_bundle_dpath, config):
    from geowatch.mlops.pipeline_nodes import ProcessNode
    from kwutil import util_parallel
    from geowatch.utils import simple_dvc
    data_workers = util_parallel.coerce_num_workers(config['data_workers'])

    model_fpaths = {
        # 'rutgers_materials': expt_dvc_dpath / 'models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth',
        # 'rutgers_materials': dvc_dpath / 'models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth',
        'rutgers_materials_model_v4': expt_dvc_dpath / 'models/rutgers/ru_model_05_25_2023.ckpt',
        'rutgers_materials_config_v4': expt_dvc_dpath / 'models/rutgers/ru_config_05_25_2023.yaml',

        'wu_mae_v1': expt_dvc_dpath / 'models/wu/wu_mae_2023_04_21/Drop6-epoch=01-val_loss=0.20.ckpt',


        # 'dzyne_s2_landcover': expt_dvc_dpath / 'models/landcover/visnav_remap_s2_subset.pt',
        'dzyne_s2_landcover': expt_dvc_dpath / 'models/landcover/sentinel2.pt',
        'dzyne_wv_landcover': expt_dvc_dpath / 'models/landcover/worldview.pt',

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

        'sam': expt_dvc_dpath / 'models/sam/sam_vit_h_4b8939.pth'
    }

    subset_name = src_fpath.name.split('.')[0]

    if subset_name.endswith('-rawbands'):
        subset_name = subset_name.rsplit('-', 1)[0]

    name_suffix = '_' + ub.hash_data(src_fpath)[0:8]

    outputs = {
        # 'rutgers_materials': aligned_bundle_dpath / (subset_name + '_rutgers_material_seg_v3' + config['kwcoco_ext']),

        'rutgers_materials_v4': aligned_bundle_dpath / (subset_name + '_rutgers_material_seg_v4' + config['kwcoco_ext']),
        'wu_mae': aligned_bundle_dpath / (subset_name + '_wu_mae' + config['kwcoco_ext']),

        'dzyne_s2_landcover': aligned_bundle_dpath / (subset_name + '_dzyne_s2_landcover' + config['kwcoco_ext']),
        'dzyne_wv_landcover': aligned_bundle_dpath / (subset_name + '_dzyne_wv_landcover' + config['kwcoco_ext']),
        'dzyne_depth': aligned_bundle_dpath / (subset_name + '_dzyne_depth' + config['kwcoco_ext']),
        'uky_invariants': aligned_bundle_dpath / (subset_name + '_uky_invariants' + config['kwcoco_ext']),
        'cold': aligned_bundle_dpath / (subset_name + '_cold' + config['kwcoco_ext']),
        'sam': aligned_bundle_dpath / (subset_name + '_sam' + config['kwcoco_ext']),
    }

    # print('Exist check: ')
    # print('model_packages: ' + ub.urepr(ub.map_vals(lambda x: x.exists(), model_fpaths)))
    # print('feature outputs: ' + ub.urepr(ub.map_vals(lambda x: x.exists(), outputs)))

    # TODO: different versions of features need different codes.
    codes = {
        'with_s2_landcover': 'LS2',
        'with_wv_landcover': 'LWV',
        'with_depth': 'D',
        'with_materials': 'M',
        'with_mae': 'E',
        'with_invariants2': 'I2',
        'with_cold': 'C',
        'with_sam': 'S',
    }

    # tmux queue is still limited. The order of submission matters.
    feature_nodes = []

    combo_code_parts = []

    key = 'with_s2_landcover'
    if config[key]:
        if config.check:
            simple_dvc.SimpleDVC().request(model_fpaths['dzyne_s2_landcover'])
        # Landcover is fairly fast to run
        node = ProcessNode(
            name=key + name_suffix,
            executable='python -m geowatch.tasks.landcover.predict',
            in_paths={
                'dataset': src_fpath,
                'deployed': model_fpaths['dzyne_s2_landcover'],
            },
            out_paths={
                'output': outputs['dzyne_s2_landcover']
            },
            algo_params={
                'with_hidden': config.num_s2_landcover_hidden,
                'select_images': '.sensor_coarse == "S2"',
                'assets_dname': config.assets_dname,
            },
            perf_params={
                'device': 0,
                'num_workers': data_workers,
            },
            node_dpath='.',
        )
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    key = 'with_wv_landcover'
    if config[key]:
        if config.check:
            simple_dvc.SimpleDVC().request(model_fpaths['dzyne_wv_landcover'])
        # Landcover is fairly fast to run
        node = ProcessNode(
            name=key + name_suffix,
            executable='python -m geowatch.tasks.landcover.predict',
            in_paths={
                'dataset': src_fpath,
                'deployed': model_fpaths['dzyne_wv_landcover'],
            },
            out_paths={
                'output': outputs['dzyne_wv_landcover']
            },
            algo_params={
                'with_hidden': config.num_wv_landcover_hidden,
                'select_images': '.sensor_coarse == "WV"',
                'assets_dname': config.assets_dname,
            },
            perf_params={
                'device': 0,
                'num_workers': data_workers,
            },
            node_dpath='.',
        )
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    key = 'with_cold'
    if config[key]:
        node = ProcessNode(
            name=key + name_suffix,
            executable='python -m geowatch.tasks.cold.predict',
            in_paths={
                'coco_fpath': src_fpath,
            },
            out_paths={
                'mod_coco_fpath': outputs['cold'],
                'out_dpath': src_fpath.parent,
            },
            algo_params={
                'sensors': 'L8',
                'adj_cloud': False,
                'method': 'COLD',
                'prob': 0.99,
                'conse': 6,
                'cm_interval': 60,
                'year_lowbound': None,
                'year_highbound': None,
                'coefs': 'cv,rmse,a0,a1,b1,c1',
                'coefs_bands': '0,1,2,3,4,5',
                'timestamp': False,
                'combine': False,
                'resolution': '30GSD',
            },
            perf_params={
                'workermode': config.cold_workermode,
                'workers': config.cold_workers,
            },
            node_dpath='.',
        )
        WITH_S2 = 1  # hard coded
        if WITH_S2:
            node.algo_params.update({
                'sensors': 'L8,S2',
                'conse': 8,
                'resolution': '10GSD',
            })
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    key = 'with_depth'
    if config[key]:
        if config.check:
            simple_dvc.SimpleDVC().request(model_fpaths['dzyne_depth'])

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

        depth_window_size = 1440
        node = ProcessNode(
            name=key + name_suffix,
            executable='python -m geowatch.tasks.depth.predict',
            in_paths={
                'dataset': src_fpath,
                'deployed': model_fpaths['dzyne_depth'],
            },
            out_paths={
                'output': outputs['dzyne_depth'],
            },
            algo_params={
                'window_size': depth_window_size,
            },
            perf_params={
                # 'skip_existing': 1,
                'data_workers': depth_data_workers,
                # 'workers': config.cold_workers,
            },
            node_dpath='.',
        )
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    key = 'with_materials'
    if config[key]:
        if config.check:
            simple_dvc.SimpleDVC().request(model_fpaths['rutgers_materials_model_v4'])
        node = ProcessNode(
            name=key + name_suffix,
            executable='python -m geowatch.tasks.rutgers_material_seg_v2.predict',
            in_paths={
                'kwcoco_fpath': src_fpath,
                'model_fpath': model_fpaths['rutgers_materials_model_v4'],
                'config_fpath': model_fpaths['rutgers_materials_config_v4'],
            },
            out_paths={
                'output_kwcoco_fpath': outputs['rutgers_materials_v4'],
            },
            algo_params={
            },
            perf_params={
                'workers': data_workers,
            },
            node_dpath='.',
        )
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    key = 'with_mae'
    if config[key]:
        if config.check:
            simple_dvc.SimpleDVC().request(model_fpaths['wu_mae_v1'])
        node = ProcessNode(
            name=key + name_suffix,
            executable=ub.codeblock(
                '''
                python -m geowatch.tasks.mae.predict
                '''),
            in_paths={
                'input_kwcoco': src_fpath,
                'mae_ckpt_path': model_fpaths['wu_mae_v1'],
            },
            out_paths={
                'output_kwcoco': outputs['wu_mae'],
            },
            algo_params={
                'assets_dname': config.assets_dname,
            },
            perf_params={
                'workers': data_workers,
            },
            node_dpath='.',
        )
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    key = 'with_invariants2'
    if config[key]:
        if config.check:
            simple_dvc.SimpleDVC().request(model_fpaths['uky_pretext2'])
        if not model_fpaths['uky_pretext2'].exists():
            print('Warning: UKY pretext model does not exist')

        # task['gpus'] = 1
        # all_tasks = 'before_after segmentation pretext'
        node = ProcessNode(
            name=key + name_suffix,
            executable=ub.codeblock(
                '''
                python -m geowatch.tasks.invariants.predict
                '''),
            in_paths={
                'input_kwcoco': src_fpath,
                'pretext_package_path': model_fpaths['uky_pretext2'],
                'pca_projection_path': model_fpaths['uky_pca'],
            },
            out_paths={
                'output_kwcoco': outputs['uky_invariants'],
            },
            algo_params={
                'assets_dname': config.assets_dname,
                'input_resolution': config['invariant_resolution'],
                'window_resolution': config['invariant_resolution'],
                'patch_size': 256,
                'patch_overlap': 0.3,
                'do_pca': config['invariant_pca'],
                'tasks': ['before_after', 'pretext'],
            },
            perf_params={
                'workers': data_workers,
                'io_workers': 0,
            },
            node_dpath='.',
        )
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    key = 'with_sam'
    if config[key]:
        if config.check:
            simple_dvc.SimpleDVC().request(model_fpaths['sam'])
        if not model_fpaths['sam'].exists():
            print('Warning: SAM model does not exist')
        node = ProcessNode(
            name=key + name_suffix,
            executable=ub.codeblock(
                '''
                python -m geowatch.tasks.sam.predict
                '''),
            in_paths={
                'input_kwcoco': src_fpath,
                'weights_fpath': model_fpaths['sam'],
            },
            out_paths={
                'output_kwcoco': outputs['sam'],
            },
            algo_params={
                'assets_dname': config.assets_dname,
                'window_overlap': 0.3,
            },
            perf_params={
                'data_workers': data_workers,
                'io_workers': 0,
            },
            node_dpath='.',
        )
        feature_nodes.append(node)
        combo_code_parts.append(codes[key])

    # Determine what all of the output paths will be
    feature_paths = []
    feature_output_nodes = []
    for node in feature_nodes:
        node_features = []
        for output in node.outputs.values():
            if output.name == 'out_dpath':
                # hack to skip a non-feature output for COLD
                continue
            node_features.append(str(output.final_value))
            feature_output_nodes.append(output)
        assert len(node_features) == 1, (
            'code assumes each node should have 1 feature output')
        feature_paths.extend(node_features)

    # Finalize features by combining them all into combo.kwcoco.json
    tocombine = [str(src_fpath)] + feature_paths
    combo_code = ''.join(sorted(combo_code_parts))

    base_combo_fpath = aligned_bundle_dpath / (f'combo_{subset_name}_{combo_code}' + config['kwcoco_ext'])

    for node in feature_nodes:
        node.configure(cache=False)

    combine_node = ProcessNode(
        name='combine_features' + name_suffix,
        executable='python -m geowatch.cli.coco_combine_features',
        in_paths={
            'src': tocombine,
        },
        out_paths={
            'dst': base_combo_fpath,
        },
    )

    # TODO: it would be nice if the mlops DAG allowed us to simply specify the
    # process level dependencies and assume we take care of the i/o level
    # dependencies.
    for output in feature_output_nodes:
        output.connect(combine_node.inputs['src'])
    combine_node.configure(cache=False)

    nodes = [combine_node] + feature_nodes
    return nodes, base_combo_fpath


main = prep_feats

if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(geowatch_dvc)
        python -m geowatch.cli.prepare_teamfeats \
            --src_kwcocos="$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json" \
            --gres=0 \
            --with_depth=0 \
            --run=False --skip_existing=False --virtualenv_cmd "conda activate geowatch" \
            --backend=serial

        python -m geowatch.cli.prepare_teamfeats --gres=0,2 --with_depth=True --keep_sessions=True
        python -m geowatch.cli.prepare_teamfeats --gres=2 --with_materials=False --keep_sessions=True

        # TODO: rename to schedule teamfeatures

        # TO UPDATE ANNOTS
        # Update to whatever the state of the annotations submodule is
        DVC_DPATH=$(geowatch_dvc)
        python -m geowatch reproject_annotations \
            --src $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --site_models="$DVC_DPATH/annotations/site_models/*.geojson"

        kwcoco stats $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data_20220203.kwcoco.json $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json

        # Team Features on Drop2
        DVC_DPATH=$(geowatch_dvc)
        python -m geowatch.cli.prepare_teamfeats \
            --src_kwcocos=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --gres=0,1 --with_depth=0 --with_materials=False  \
            --run=0

        ###
        DATASET_CODE=Aligned-Drop2-TA1-2022-02-24
        DVC_DPATH=$(geowatch_dvc)
        DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
        python -m geowatch.cli.prepare_teamfeats \
            --src_kwcocos=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --gres=0,1 \
            --with_depth=1 \
            --with_s2_landcover=1 \
            --with_materials=1 \
            --depth_workers=auto \
            --skip_existing=0 --run=0

        # Simple demo
        python -m geowatch.cli.prepare_teamfeats \
            --src_kwcocos=./mydata/data.kwcoco.json \
            --gres=0,1 \
            --with_depth=0 \
            --with_s2_landcover=1 \
            --with_materials=1 \
            --skip_existing=0 \
            --backend=tmux \
            --run=0


    """
    main(cmdline=True)

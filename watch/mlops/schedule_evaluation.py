"""
Helper for scheduling a set of prediction + evaluation jobs

TODO:
    - [ ] Differentiate between pixel models for different tasks.
    - [ ] Allow the output of tracking to feed into activity classification
"""
import ubelt as ub
import shlex
import json
import scriptconfig as scfg
# import watch
import shlex  # NOQA
import json  # NOQA
from watch.utils.lightning_ext import util_globals  # NOQA
import kwarray  # NOQA
import cmd_queue
import networkx as nx
import itertools as it
from watch.utils.util_param_grid import expand_param_grid, dotdict_to_nested


class ScheduleEvaluationConfig(scfg.Config):
    """
    Builds commands and optionally schedules them.
    """
    default = {
        'devices': scfg.Value('auto', help='if using tmux or serial, indicate which gpus are available for use as a comma separated list: e.g. 0,1'),
        'run': scfg.Value(False, help='if False, only prints the commands, otherwise executes them'),
        'virtualenv_cmd': scfg.Value(None, help='command to activate a virtualenv if needed. (might have issues with slurm backend)'),
        'skip_existing': scfg.Value(False, help='if True dont submit commands where the expected products already exist'),
        'backend': scfg.Value('tmux', help='can be tmux, slurm, or maybe serial in the future'),

        'pred_workers': scfg.Value(4, help='number of prediction workers in each process'),

        'shuffle_jobs': scfg.Value(True, help='if True, shuffles the jobs so they are submitted in a random order'),
        'annotations_dpath': scfg.Value(None, help='path to IARPA annotations dpath for IARPA eval'),

        'expt_dvc_dpath': None,
        'data_dvc_dpath': None,

        'check_other_sessions': scfg.Value('auto', help='if True, will ask to kill other sessions that might exist'),
        'queue_size': scfg.Value('auto', help='if auto, defaults to number of GPUs'),

        # These enabled flags should probably be pushed off to params
        'enable_pred_trk_pxl': scfg.Value(True, isflag=True, help='BAS heatmap'),
        'enable_pred_trk_poly': scfg.Value(True, isflag=True, help='BAS tracking'),
        'enable_crop': scfg.Value(True, isflag=True, help='SC tracking'),
        'enable_pred_act_pxl': scfg.Value(True, isflag=True, help='SC heatmaps'),
        'enable_pred_act_poly': scfg.Value(True, isflag=True, help='SC tracking'),

        'enable_eval_trk_pxl': scfg.Value(True, isflag=True, help='BAS heatmap evaluation'),
        'enable_eval_trk_poly': scfg.Value(True, isflag=True, help='BAS tracking evaluation'),
        'enable_eval_act_pxl': scfg.Value(True, isflag=True, help='SC heatmaps evaluation'),
        'enable_eval_act_poly': scfg.Value(True, isflag=True, help='SC tracking evaluation'),
        'enable_viz_pred_trk_poly': scfg.Value(False, isflag=True, help='if true draw predicted tracks'),

        'draw_heatmaps': scfg.Value(1, isflag=True, help='if true draw heatmaps on eval'),
        'draw_curves': scfg.Value(1, isflag=True, help='if true draw curves on eval'),

        'partition': scfg.Value(None, help='specify slurm partition (slurm backend only)'),
        'mem': scfg.Value(None, help='specify slurm memory per task (slurm backend only)'),

        'params': scfg.Value(None, type=str, help='a yaml/json grid/matrix of prediction params'),
    }


def schedule_evaluation(cmdline=False, **kwargs):
    r"""
    First ensure that models have been copied to the DVC repo in the
    appropriate path. (as noted by model_dpath)

    Ignore:
        from watch.mlops.schedule_evaluation import *  # NOQA
        kwargs = {'params': ub.codeblock(
            '''
            - matrix:
                ###
                ### BAS Pixel Prediction
                ###
                trk.pxl.model:
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=73-step=37888.pt.pt
                trk.pxl.data.test_dataset:
                    - ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_bas.json
                trk.pxl.data.tta_time: 0
                trk.pxl.data.chip_overlap: 0.3
                trk.pxl.data.window_scale_space: 10GSD
                trk.pxl.data.input_scale_space: 15GSD
                trk.pxl.data.output_scale_space: 15GSD
                trk.pxl.data.time_span: auto
                trk.pxl.data.time_sampling: auto
                trk.pxl.data.time_steps: auto
                trk.pxl.data.chip_dims: auto
                trk.pxl.data.set_cover_algo: None
                trk.pxl.data.resample_invalid_frames: 1
                trk.pxl.data.use_cloudmask: 1
                ###
                ### BAS Polygon Prediction
                ###
                trk.poly.thresh: [0.10, 0.15]
                trk.poly.morph_kernel: 3
                trk.poly.norm_ord: 1
                trk.poly.agg_fn: probs
                trk.poly.thresh_hysteresis: None
                trk.poly.moving_window_size: None
                trk.poly.polygon_fn: heatmaps_to_polys
                ###
                ### SC Pixel Prediction
                ###
                crop.src: ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/kwcoco_for_sc.json
                crop.regions:
                    - trk.poly.output
                    - truth
                act.pxl.data.test_dataset:
                    - crop.dst
                    # - ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_sc.json
                act.pxl.model:
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
            ''')}
        cmdline = 0
    """
    import watch
    config = ScheduleEvaluationConfig(cmdline=cmdline, data=kwargs)
    print('ScheduleEvaluationConfig config = {}'.format(ub.repr2(dict(config), nl=1, si=1)))

    expt_dvc_dpath = config['expt_dvc_dpath']
    data_dvc_dpath = config['data_dvc_dpath']
    if expt_dvc_dpath is None:
        expt_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_expt')
    if data_dvc_dpath is None:
        data_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_data')
    data_dvc_dpath = ub.Path(data_dvc_dpath)
    expt_dvc_dpath = ub.Path(expt_dvc_dpath)

    from watch.mlops.expt_manager import ExperimentState
    # start using the experiment state logic as the path and metadata
    # organization logic
    state = ExperimentState(expt_dvc_dpath, '*')

    # Get truth annotations
    annotations_dpath = config['annotations_dpath']
    if annotations_dpath is None:
        annotations_dpath = data_dvc_dpath / 'annotations'
    annotations_dpath = ub.Path(annotations_dpath)
    region_model_dpath = annotations_dpath / 'region_models'

    # Expand paramater search grid
    # arg = config['params']
    all_param_grid = expand_param_grid(config['params'])

    grid_item_defaults = ub.udict({
        'trk.pxl.model': None,
        'trk.pxl.data.test_dataset': None,
        'trk.poly.thresh': 0.1,

        'crop.src': None,
        'crop.context_factor': 1.5,
        'crop.regions': 'truth',

        'act.pxl.model': None,
        'act.pxl.data.test_dataset': None,
        'act.poly.thresh': 0.1,
    })

    resolved_rows = []
    # Resolve parameters for each row
    for item in all_param_grid:
        row = resolve_pipeline_row(grid_item_defaults, state,
                                   region_model_dpath, expt_dvc_dpath, item)
        resolved_rows.append(row)

    # from rich import print

    queue_dpath = expt_dvc_dpath / '_cmd_queue_schedule'
    queue_dpath.ensuredir()

    devices = config['devices']
    print('devices = {!r}'.format(devices))
    if devices == 'auto':
        GPUS = _auto_gpus()
    else:
        GPUS = None if devices is None else ensure_iterable(devices)
    print('GPUS = {!r}'.format(GPUS))

    queue_size = config['queue_size']
    if queue_size == 'auto':
        queue_size = len(GPUS)

    common_submitkw = dict(
        partition=config['partition'],
        mem=config['mem']
    )
    environ = {}
    queue = cmd_queue.Queue.create(config['backend'], name='schedule-eval',
                                   size=queue_size, environ=environ,
                                   dpath=queue_dpath, gres=GPUS)

    virtualenv_cmd = config['virtualenv_cmd']
    if virtualenv_cmd:
        queue.add_header_command(virtualenv_cmd)

    # Each row represents a single source-to-sink pipeline run, but multiple
    # rows may share pipeline steps. This is handled by having unique ids per
    # job that depend on their outputs.
    for row in resolved_rows:
        paths = row['paths']
        paths['true_annotations_dpath'] = annotations_dpath
        paths['true_site_dpath'] = paths['true_annotations_dpath'] / 'site_models'
        paths['true_region_dpath'] = paths['true_annotations_dpath'] / 'region_models'
        task_params = row['task_params']
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())

        trk_pxl_params = task_params['trk.pxl']
        trk_poly_params = task_params['trk.poly']
        crop_params = task_params['crop']
        act_pxl_params = task_params['act.pxl']
        act_poly_params = task_params['act.poly']
        condensed = row['condensed']

        ### define the dag of this row item.

        steps = {}

        step = Pipeline.pred_trk_pxl(trk_pxl_params, **paths)
        steps[step.name] = step

        step = Pipeline.pred_trk_poly(trk_poly_params, **paths)
        steps[step.name] = step

        step = Pipeline.act_crop(crop_params, **paths)
        steps[step.name] = step

        step = Pipeline.pred_act_pxl(act_pxl_params, **paths)
        steps[step.name] = step

        step = Pipeline.pred_act_poly(act_poly_params, **paths)
        steps[step.name] = step

        step = Pipeline.eval_trk_pxl(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.eval_trk_poly(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.viz_pred_trk_poly(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.eval_act_pxl(condensed, **paths)
        steps[step.name] = step

        step = Pipeline.eval_act_poly(condensed, **paths)
        steps[step.name] = step

        # Determine the interaction / dependencies between step inputs /
        # outputs
        g = nx.DiGraph()
        outputs_to_step = ub.ddict(list)
        inputs_to_step = ub.ddict(list)
        for step in steps.values():
            for path in step.out_paths.values():
                outputs_to_step[path].append(step.name)
            for path in step.in_paths.values():
                inputs_to_step[path].append(step.name)
            g.add_node(step.name, step=step)

        inputs_to_step = ub.udict(inputs_to_step)
        outputs_to_step = ub.udict(outputs_to_step)

        common = list((inputs_to_step & outputs_to_step).keys())
        for path in common:
            isteps = inputs_to_step[path]
            osteps = outputs_to_step[path]
            for istep, ostep in it.product(isteps, osteps):
                g.add_edge(ostep, istep)

        #
        # Determine which steps are enabled / disabled

        sorted_nodes = list(nx.topological_sort(g))
        for node in sorted_nodes:
            step = g.nodes[node]['step']
            step.enabled = config['enable_' + step.name]
            # if config['skip_existing']:
            ancestors_will_exist = all(
                g.nodes[ancestor]['step'].will_exist
                for ancestor in nx.ancestors(g, step.name)
            )
            step.does_exist = all(
                step.out_paths.map_values(lambda p: p.exists()).values()
            )
            if config['skip_existing'] and step.does_exist:
                step.enabled = False
            step.will_exist = (
                (step.enabled and ancestors_will_exist) or
                step.does_exist
            )
            print('step.out_paths = {}'.format(ub.repr2(step.out_paths, nl=1)))
            print(f'{step.name=}, does_exist={step.does_exist} will_exist={step.will_exist} enabled={step.enabled}')

        #
        # Submit steps to the scheduling queue
        for node in sorted_nodes:
            # Skip duplicate jobs
            step = g.nodes[node]['step']
            if step.node_id in queue.named_jobs:
                continue
            depends = []
            for other, _ in list(g.in_edges(node)):
                dep_step = g.nodes[other]['step']
                if dep_step.enabled:
                    depends.append(dep_step.node_id)

            if step.will_exist and step.enabled:
                queue.submit(command=step.command, name=step.node_id,
                             depends=depends, **common_submitkw)

    print('queue = {!r}'.format(queue))
    # print(f'{len(queue)=}')
    with_status = 0
    with_rich = 0
    queue.rprint(with_status=with_status, with_rich=with_rich)

    queue.write_network_text()

    for job in queue.jobs:
        # TODO: should be able to set this as a queue param.
        job.log = False

    # RUN
    if config['run']:
        # ub.cmd('bash ' + str(driver_fpath), verbose=3, check=True)
        queue.run(block=True, check_other_sessions=config['check_other_sessions'])
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


def resolve_pipeline_row(grid_item_defaults, state, region_model_dpath, expt_dvc_dpath, item):
    from watch.mlops.expt_manager import ExperimentState
    state = ExperimentState(expt_dvc_dpath, '*')
    item = grid_item_defaults | item
    print('item = {}'.format(ub.repr2(item, nl=1)))
    nested = dotdict_to_nested(item)

    condensed = {}
    paths = {}

    # Might not need this exactly
    pkg_trk_pixel_pathcfg = state._parse_pattern_attrs(
        state.templates['pkg_trk_pxl_fpath'], item['trk.pxl.model'])
    # fixme: dataset code is ambiguous between BAS and SC
    # pkg_trk_pixel_pathcfg.pop('dataset_code', None)
    condensed.update(pkg_trk_pixel_pathcfg)

    condensed['expt_dvc_dpath'] = expt_dvc_dpath

    ### BAS / TRACKING ###

    trk_pxl  = nested['trk']['pxl']
    trk_poly = nested['trk']['poly']
    trk_pxl_params = ub.udict(trk_pxl['data']) - {'test_dataset'}
    trk_poly_params = ub.udict(trk_poly)

    condensed['trk_model'] = state._condense_model(item['trk.pxl.model'])
    condensed['test_trk_dset'] = state._condense_test_dset(item['trk.pxl.data.test_dataset'])
    condensed['trk_pxl_cfg'] = state._condense_cfg(trk_pxl_params, 'trk_pxl')
    condensed['trk_poly_cfg'] = state._condense_cfg(trk_poly_params, 'trk_poly')

    paths['pkg_trk_pxl_fpath'] = ub.Path(item['trk.pxl.model'])
    paths['trk_test_dataset_fpath'] = item['trk.pxl.data.test_dataset']
    paths['pred_trk_pxl_fpath'] = ub.Path(state.templates['pred_trk_pxl_fpath'].format(**condensed))
    paths['eval_trk_pxl_dpath'] = ub.Path(state.templates['eval_trk_pxl_dpath'].format(**condensed))
    paths['eval_trk_pxl_fpath'] = ub.Path(state.templates['eval_trk_pxl_fpath'].format(**condensed))
    paths['pred_trk_poly_fpath'] = ub.Path(state.templates['pred_trk_poly_fpath'].format(**condensed))
    paths['pred_trk_poly_viz_stamp'] = ub.Path(state.templates['pred_trk_poly_viz_stamp'].format(**condensed))
    paths['pred_trk_poly_dpath'] = ub.Path(state.templates['pred_trk_poly_dpath'].format(**condensed))
    paths['pred_trk_poly_kwcoco'] = ub.Path(state.templates['pred_trk_poly_kwcoco'].format(**condensed))
    paths['eval_trk_poly_fpath'] = ub.Path(state.templates['eval_trk_poly_fpath'].format(**condensed))
    paths['eval_trk_poly_dpath'] = ub.Path(state.templates['eval_trk_poly_dpath'].format(**condensed))

    trackid_deps = {}
    trackid_deps = ub.udict(condensed) & (
        {'trk_model', 'test_trk_dset', 'trk_pxl_cfg', 'trk_poly_cfg'})
    condensed['trk_poly_id'] = state._condense_cfg(trackid_deps, 'trk_poly_id')

    ### CROPPING ###

    crop_params = ub.udict(nested['crop']).copy()
    crop_src_fpath = crop_params.pop('src')
    paths['crop_src_fpath'] = crop_src_fpath
    if crop_params['regions'] == 'truth':
        # Crop job depends only on true annotations
        paths['crop_regions'] = str(region_model_dpath) + '/*.geojson'
        condensed['regions_id'] = 'truth'  # todo: version info
    if crop_params['regions'] == 'trk.poly.output':
        # Crop job depends on track predictions
        paths['crop_regions'] = state.templates['pred_trk_poly_fpath'].format(**condensed)
        condensed['regions_id'] = condensed['trk_poly_id']

    crop_params['regions'] = paths['crop_regions']
    condensed['crop_cfg'] = state._condense_cfg(crop_params, 'crop')
    condensed['crop_src_dset'] = state._condense_test_dset(crop_src_fpath)

    crop_id_deps = ub.udict(condensed) & (
        {'regions_id', 'crop_cfg', 'crop_src_dset'})
    condensed['crop_id'] = state._condense_cfg(crop_id_deps, 'crop_id')
    paths['crop_dpath'] = ub.Path(state.templates['crop_dpath'].format(**condensed))
    paths['crop_fpath'] = ub.Path(state.templates['crop_fpath'].format(**condensed))
    condensed['crop_dst_dset'] = state._condense_test_dset(paths['crop_fpath'])

    ### SC / ACTIVITY ###
    act_pxl  = nested['act']['pxl']
    act_poly = nested['act']['poly']
    act_pxl_params = ub.udict(act_pxl['data']) - {'test_dataset'}
    act_poly_params = ub.udict(act_poly)

    condensed['act_model'] = state._condense_model(item['act.pxl.model'])
    condensed['act_pxl_cfg'] = state._condense_cfg(act_pxl_params, 'act_pxl')
    condensed['act_poly_cfg'] = state._condense_cfg(act_poly_params, 'act_poly')

    pkg_act_pixel_pathcfg = state._parse_pattern_attrs(state.templates['pkg_act_pxl_fpath'], item['act.pxl.model'])
    condensed.update(pkg_act_pixel_pathcfg)

    # paths['act_test_dataset_fpath'] = item['act.pxl.data.test_dataset']
    if item['act.pxl.data.test_dataset'] == 'crop.dst':
        # Activity prediction depends on a cropping job
        paths['act_test_dataset_fpath'] = paths['crop_fpath']
    else:
        # Activity prediction has no dependencies in this case.
        paths['act_test_dataset_fpath'] = item['act.pxl.data.test_dataset']
    condensed['test_act_dset'] = state._condense_test_dset(paths['act_test_dataset_fpath'])

    paths['pkg_act_pxl_fpath'] = ub.Path(item['act.pxl.model'])
    paths['pred_act_poly_fpath'] = ub.Path(state.templates['pred_act_poly_fpath'].format(**condensed))
    paths['pred_act_poly_dpath'] = ub.Path(state.templates['pred_act_poly_dpath'].format(**condensed))
    paths['pred_act_pxl_fpath'] = ub.Path(state.templates['pred_act_pxl_fpath'].format(**condensed))
    paths['eval_act_pxl_fpath'] = ub.Path(state.templates['eval_act_pxl_fpath'].format(**condensed))
    paths['eval_act_pxl_dpath'] = ub.Path(state.templates['eval_act_pxl_dpath'].format(**condensed))
    paths['eval_act_poly_fpath'] = ub.Path(state.templates['eval_act_poly_fpath'].format(**condensed))
    paths['eval_act_poly_dpath'] = ub.Path(state.templates['eval_act_poly_dpath'].format(**condensed))
    paths['pred_act_poly_kwcoco'] = ub.Path(state.templates['pred_act_poly_kwcoco'].format(**condensed))

    # paths['eval_act_tmp_dpath'] = pred_act_row['eval_act_poly_dpath'] / '_tmp'

    task_params = {
        'trk.pxl': trk_pxl_params,
        'trk.poly': trk_poly_params,
        'crop': crop_params,
        'act.pxl': act_pxl_params,
        'act.poly': act_poly_params,
    }

    row = {
        'condensed': condensed,
        'paths': paths,
        'item': item,
        'task_params': task_params,
    }
    return row


def resolve_package_paths(model_globstr, expt_dvc_dpath):
    import rich
    # import glob
    from watch.utils import util_pattern

    # HACK FOR DVC PTH FIXME:
    # if str(model_globstr).endswith('.txt'):
    #     from watch.utils.simple_dvc import SimpleDVC
    #     print('model_globstr = {!r}'.format(model_globstr))
    #     # if expt_dvc_dpath is None:
    #     #     expt_dvc_dpath = SimpleDVC.find_root(ub.Path(model_globstr))

    def expand_model_list_file(model_lists_fpath, expt_dvc_dpath=None):
        """
        Given a file containing paths to models, expand it into individual
        paths.
        """
        expanded_fpaths = []
        lines = [line for line in ub.Path(model_globstr).read_text().split('\n') if line]
        missing = []
        for line in lines:
            if expt_dvc_dpath is not None:
                package_fpath = ub.Path(expt_dvc_dpath / line)
            else:
                package_fpath = ub.Path(line)
            if package_fpath.is_file():
                expanded_fpaths.append(package_fpath)
            else:
                missing.append(line)
        if missing:
            rich.print('[yellow] WARNING: missing = {}'.format(ub.repr2(missing, nl=1)))
            rich.print(f'[yellow] WARNING: specified a models-of-interest.txt and {len(missing)} / {len(lines)} models were missing')
        return expanded_fpaths

    print('model_globstr = {!r}'.format(model_globstr))
    model_globstr = util_pattern.MultiPattern.coerce(model_globstr)
    package_fpaths = []
    # for package_fpath in glob.glob(model_globstr, recursive=True):
    for package_fpath in model_globstr.paths(recursive=True):
        package_fpath = ub.Path(package_fpath)
        if package_fpath.name.endswith('.txt'):
            # HACK FOR PATH OF MODELS
            model_lists_fpath = package_fpath
            expanded_fpaths = expand_model_list_file(model_lists_fpath, expt_dvc_dpath=expt_dvc_dpath)
            package_fpaths.extend(expanded_fpaths)
        else:
            package_fpaths.append(package_fpath)

    if len(package_fpaths) == 0:
        import pathlib
        if '*' not in str(model_globstr):
            package_fpaths = [ub.Path(model_globstr)]
        elif isinstance(model_globstr, (str, pathlib.Path)):
            # Warn the user if they gave a bad model globstr (this is just one
            # of the many potential ways things could go wrong)
            glob_path = ub.Path(model_globstr)
            def _concrete_glob_part(path):
                " Find the resolved part of the glob path "
                concrete_parts = []
                for p in path.parts:
                    if '*' in p:
                        break
                    concrete_parts.append(p)
                return ub.Path(*concrete_parts)
            concrete = _concrete_glob_part(glob_path)
            if not concrete.exists():
                rich.print('[yellow] WARNING: part of the model_globstr does not exist: {}'.format(concrete))

    return package_fpaths


class Step:
    def __init__(step, name, command, in_paths, out_paths, resources):
        step.name = name
        step.command = command
        step.in_paths = in_paths
        step.out_paths = out_paths
        step.resources = resources
        #
        # Set later
        step.enabled = None
        step.will_exist = None

    @property
    def node_id(step):
        """
        The experiment manager constructs output paths such that they
        are unique given the specific set of inputs and parameters. Thus
        the output paths are sufficient to determine a unique id per step.
        """
        return step.name + '_' + ub.hash_data(step.out_paths)[0:12]

    def test_is_computed_command(step):
        test_cmds = ['test -f "{p}"' for p in step.out_paths.values()]
        if len(test_cmds) == 1:
            return test_cmds[0]
        else:
            return '(' + ' && '.join(test_cmds) + ')'


class Pipeline:
    """
    Registers how to call each step in the pipeline
    """

    def __init__(pipe):
        pass

    @staticmethod
    def _make_argstr(params):
        parts = [f'    --{k}={v} \\' for k, v in params.items()]
        return chr(10).join(parts).lstrip().rstrip('\\')

    def act_crop(crop_params, **paths):
        paths = ub.udict(paths)
        from watch.cli import coco_align_geotiffs
        confobj = coco_align_geotiffs.__config__
        known_args = set(confobj.default.keys())
        assert not len(ub.udict(crop_params) - known_args), 'unknown args'
        crop_params = {
            'geo_preprop': 'auto',
            'keep': 'img',
            'force_nodata': -9999,
            'rpc_align_method': 'orthorectify',
            'target_gsd': 4,
        } | ub.udict(crop_params)
        perf_options = {
            'verbose': 1,
            'workers': 8,
            'aux_workers': 8,
            'debug_valid_regions': False,
            'visualize': False,
        }
        crop_kwargs = { **paths }
        crop_kwargs['crop_params_argstr'] = Pipeline._make_argstr(crop_params)
        crop_kwargs['crop_perf_argstr'] = Pipeline._make_argstr(perf_options)
        command = ub.codeblock(
            r'''
            python -m watch.cli.coco_align_geotiffs \
                --src "{crop_src_fpath}" \
                --dst "{crop_fpath}" \
                {crop_params_argstr} \
                {crop_perf_argstr}
            ''').format(**crop_kwargs)
        name = 'crop'
        step = Step(name, command,
                    in_paths=paths & {
                        'crop_regions',
                        'crop_src_fpath',
                    },
                    out_paths=paths & {'crop_fpath'},
                    resources={
                        'cpus': 2,
                    })
        return step

    def pred_trk_pxl(trk_pxl_params, **paths):
        paths = ub.udict(paths)
        workers = 4  # todo: parametarize
        perf_options = {
            'num_workers': workers,
            'devices': '0,',
            'accelerator': 'gpu',
            'batch_size': 1,
        }
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        pred_trk_pxl_kw =  { **paths }
        pred_trk_pxl_kw['params_argstr'] = Pipeline._make_argstr(trk_pxl_params)
        pred_trk_pxl_kw['perf_argstr'] = Pipeline._make_argstr(perf_options)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.predict \
                --package_fpath={pkg_trk_pxl_fpath} \
                --test_dataset={trk_test_dataset_fpath} \
                --pred_dataset={pred_trk_pxl_fpath} \
                {params_argstr} \
                {perf_argstr}
            ''').format(**pred_trk_pxl_kw)
        name = 'pred_trk_pxl'
        step = Step(name, command,
                    in_paths=paths & {'pkg_trk_pxl_fpath',
                                      'trk_test_dataset_fpath'},
                    out_paths=paths & {'pred_trk_pxl_fpath'},
                    resources={
                        'cpus': workers,
                        'gpus': 2,
                    })
        return step

    def pred_act_pxl(act_pxl_params, **paths):
        paths = ub.udict(paths)
        workers = 4
        perf_options = {
            'num_workers': workers,
            'devices': '0,',
            'accelerator': 'gpu',
            'batch_size': 1,
        }
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        pred_act_pxl_kw =  { **paths }
        pred_act_pxl_kw['params_argstr'] = Pipeline._make_argstr(act_pxl_params)
        pred_act_pxl_kw['perf_argstr'] = Pipeline._make_argstr(perf_options)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.predict \
                --package_fpath={pkg_act_pxl_fpath} \
                --test_dataset={act_test_dataset_fpath} \
                --pred_dataset={pred_act_pxl_fpath} \
                {params_argstr} \
                {perf_argstr}
            ''').format(**pred_act_pxl_kw)
        name = 'pred_act_pxl'
        step = Step(name, command,
                    in_paths=paths & {
                        'pkg_act_pxl_fpath',
                        'act_test_dataset_fpath'
                    },
                    out_paths=paths & {'pred_act_pxl_fpath'},
                    resources={
                        'cpus': workers,
                        'gpus': 2,
                    })
        return step

    def pred_trk_poly(trk_poly_params, **paths):
        paths = ub.udict(paths)
        pred_trk_poly_kw = { **paths }
        cfg = trk_poly_params.copy()
        if 'thresh_hysteresis' in cfg:
            if isinstance(cfg['thresh_hysteresis'], str):
                cfg['thresh_hysteresis'] = util_globals.restricted_eval(
                    cfg['thresh_hysteresis'].format(**cfg))

        if 'moving_window_size' in cfg:
            if isinstance(cfg['moving_window_size'], str):
                cfg['moving_window_size'] = util_globals.restricted_eval(
                    cfg['moving_window_size'].format(**cfg))
        else:
            cfg['moving_window_size'] = None

        if cfg['moving_window_size'] is None:
            cfg['polygon_fn'] = 'heatmaps_to_polys'
        else:
            cfg['polygon_fn'] = 'heatmaps_to_polys_moving_window'
        # kwargs['params_argstr'] = Pipeline(trk_poly_params)
        pred_trk_poly_kw['track_kwargs_str'] = shlex.quote(json.dumps(cfg))

        command = ub.codeblock(
            r'''
            python -m watch.cli.run_tracker \
                "{pred_trk_pxl_fpath}" \
                --default_track_fn saliency_heatmaps \
                --track_kwargs {track_kwargs_str} \
                --clear_annots \
                --out_dir "{pred_trk_poly_dpath}" \
                --out_fpath "{pred_trk_poly_fpath}" \
                --out_kwcoco "{pred_trk_poly_kwcoco}"
            ''').format(**pred_trk_poly_kw)
        name = 'pred_trk_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_pxl_fpath'},
                    out_paths=paths & {'pred_trk_poly_fpath',
                                       'pred_trk_poly_kwcoco'},
                    resources={'cpus': 2})
        return step

    def eval_trk_pxl(condensed, **paths):
        paths = ub.udict(paths)
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        eval_trk_pxl_kwe = { **paths }
        extra_opts = {
            'draw_curves': True,  # todo: parametarize
            'draw_heatmaps': True,  # todo: parametarize
            'viz_thresh': 0.2,
            'workers': 2,
        }
        eval_trk_pxl_kwe['extra_argstr'] = Pipeline._make_argstr(extra_opts)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.evaluate \
                --true_dataset={trk_test_dataset_fpath} \
                --pred_dataset={pred_trk_pxl_fpath} \
                --eval_dpath={eval_trk_pxl_dpath} \
                --score_space=video \
                {extra_argstr}
            ''').format(**eval_trk_pxl_kwe)
        name = 'eval_trk_pxl'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_pxl_fpath'},
                    out_paths=paths & {'eval_trk_pxl_fpath'},
                    resources={'cpus': 2})
        return step

    def viz_pred_trk_poly(condensed, **paths):
        paths = ub.udict(paths)
        viz_pred_trk_poly_kw = paths.copy()
        viz_pred_trk_poly_kw['extra_header'] = f"\\n{condensed['trk_pxl_cfg']}-{condensed['trk_poly_cfg']}"
        command = ub.codeblock(
            r'''
            smartwatch visualize \
                "{pred_trk_poly_kwcoco}" \
                --channels="red|green|blue,salient" \
                --stack=only \
                --workers=avail/2 \
                --workers=avail/2 \
                --extra_header="{extra_header}" \
                --animate=True && touch {pred_trk_poly_viz_stamp}
            ''').format(**viz_pred_trk_poly_kw)
        name = 'viz_pred_trk_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_poly_kwcoco'},
                    out_paths=paths & {'pred_trk_poly_viz_stamp'},
                    resources={'cpus': 2})
        return step

    def eval_trk_poly(condensed, **paths):
        paths = ub.udict(paths)
        eval_trk_poly_kw = { **paths }
        eval_trk_poly_kw['eval_trk_poly_dpath'] = eval_trk_poly_kw['eval_trk_poly_dpath']
        eval_trk_poly_kw['eval_trk_poly_tmp_dpath'] = eval_trk_poly_kw['eval_trk_poly_dpath'] / '_tmp'
        eval_trk_poly_kw['name_suffix'] = '-'.join([
            condensed['trk_model'],
            condensed['trk_pxl_cfg'],
            condensed['trk_poly_cfg'],
        ])
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_metrics_framework \
                --merge=True \
                --inputs_are_paths=True \
                --name "{name_suffix}" \
                --true_site_dpath "{true_site_dpath}" \
                --true_region_dpath "{true_region_dpath}" \
                --pred_sites "{pred_trk_poly_fpath}" \
                --tmp_dir "{eval_trk_poly_tmp_dpath}" \
                --out_dir "{eval_trk_poly_dpath}" \
                --merge_fpath "{eval_trk_poly_fpath}"
            ''').format(**eval_trk_poly_kw)
        name = 'eval_trk_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_trk_poly_fpath'},
                    out_paths=paths & {'eval_trk_poly_fpath'},
                    resources={'cpus': 2})
        return step

    def pred_act_poly(act_poly_params, **paths):
        paths = ub.udict(paths)
        # pred_act_row['site_summary_glob'] = (region_model_dpath / '*.geojson')
        pred_act_poly_kw = paths.copy()
        actclf_cfg = {
            'boundaries_as': 'polys',
        }
        actclf_cfg.update(act_poly_params)
        pred_act_poly_kw['kwargs_str'] = shlex.quote(json.dumps(actclf_cfg))
        # pred_act_poly_kw['site_summary_glob'] = (pred_act_poly_kw['region_model_dpath'] / '*.geojson')
        pred_act_poly_kw['site_summary_glob'] = 'TODO not well defined yet'
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_tracker \
                "{pred_act_pxl_fpath}" \
                --site_summary '{site_summary_glob}' \
                --default_track_fn class_heatmaps \
                --track_kwargs {kwargs_str} \
                --out_dir "{pred_act_poly_dpath}" \
                --out_fpath "{pred_act_poly_fpath}" \
                --out_kwcoco_fpath "{pred_act_poly_kwcoco}"
            ''').format(**pred_act_poly_kw)
        name = 'pred_act_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_act_pxl_fpath'},
                    out_paths=paths & {'pred_act_poly_fpath',
                                       'pred_act_poly_kwcoco'},
                    resources={'cpus': 2})
        return step

    def eval_act_pxl(condensed, **paths):
        paths = ub.udict(paths)
        paths = ub.udict(paths).map_values(lambda p: ub.Path(p).expand())
        eval_act_pxl_kwe = { **paths }
        extra_opts = {
            'draw_curves': True,
            'draw_heatmaps': True,
            'viz_thresh': 0.2,
            'workers': 2,
        }
        eval_act_pxl_kwe['extra_argstr'] = Pipeline._make_argstr(extra_opts)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.evaluate \
                --true_dataset={act_test_dataset_fpath} \
                --pred_dataset={pred_act_pxl_fpath} \
                --eval_dpath={eval_act_pxl_dpath} \
                --score_space=video \
                {extra_argstr}
            ''').format(**eval_act_pxl_kwe)
        name = 'eval_act_pxl'
        step = Step(name, command,
                    in_paths=paths & {'pred_act_pxl_fpath'},
                    out_paths=paths & {'eval_act_pxl_fpath'},
                    resources={'cpus': 2})
        return step

    def eval_act_poly(condensed, **paths):
        paths = ub.udict(paths)
        eval_act_poly_kw = paths.copy()
        eval_act_poly_kw['name_suffix'] = '-'.join([
            # condensed['crop_dst_dset']
            condensed['test_act_dset'],
            condensed['act_model'],
            condensed['act_pxl_cfg'],
            condensed['act_poly_cfg'],
        ])
        eval_act_poly_kw['eval_act_poly_dpath'] = eval_act_poly_kw['eval_trk_poly_dpath']
        eval_act_poly_kw['eval_act_poly_tmp_dpath'] = eval_act_poly_kw['eval_trk_poly_dpath'] / '_tmp'
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_metrics_framework \
                --merge=True \
                --inputs_are_paths=True \
                --name "{name_suffix}" \
                --true_site_dpath "{true_site_dpath}" \
                --true_region_dpath "{true_region_dpath}" \
                --pred_sites "{pred_act_poly_fpath}" \
                --tmp_dir "{eval_act_poly_tmp_dpath}" \
                --out_dir "{eval_act_poly_dpath}" \
                --merge_fpath "{eval_act_poly_fpath}" \
            ''').format(**eval_act_poly_kw)
        name = 'eval_act_poly'
        step = Step(name, command,
                    in_paths=paths & {'pred_act_poly_fpath'},
                    out_paths=paths & {'eval_act_poly_fpath'},
                    resources={'cpus': 2})
        return step


if __name__ == '__main__':
    schedule_evaluation(cmdline=True)

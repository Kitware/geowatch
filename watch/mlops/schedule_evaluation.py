r"""
Helper for scheduling a set of prediction + evaluation jobs

TODO:
    - [ ] Differentiate between pixel models for different tasks.
    - [ ] Allow the output of tracking to feed into activity classification



Example:

    python -m watch.mlops.schedule_evaluation \
        --params="
            matrix:
                trk.pxl.package_fpath:
                    - ./my_bas_model.pt
                trk.pxl.data.test_dataset:
                    - ./my_test_dataset/bas_ready_data.kwcoco.json
                trk.pxl.data.window_space_scale: 15GSD
                trk.pxl.data.time_sampling:
                    - "auto"
                trk.pxl.data.input_space_scale:
                    - "15GSD"
                trk.poly.moving_window_size:
                    - null
                crop.src:
                    - ./my_test_dataset/sc_query_data.kwcoco.json
                crop.regions:
                    - trk.poly.output
                act.pxl.data.test_dataset:
                    - crop.dst
                act.pxl.data.window_space_scale:
                    - auto
                act.poly.thresh:
                    - 0.1
                act.poly.use_viterbi:
                    - 0
                act.pxl.package_fpath:
                    - my_sc_model.pt
        " \
        --expt_dvc_dpath=./my_expt_dir \
        --data_dvc_dpath=./my_data_dir \
        --dynamic_skips=0 \
        --enable_pred_trk_pxl=1 \
        --enable_pred_trk_poly=1 \
        --enable_eval_trk_pxl=0 \
        --enable_eval_trk_poly=0 \
        --enable_crop=1 \
        --enable_pred_act_pxl=1 \
        --enable_pred_act_poly=1 \
        --enable_eval_act_pxl=0 \
        --enable_eval_act_poly=0 \
        --enable_viz_pred_trk_poly=0 \
        --enable_viz_pred_act_poly=0 \
        --enable_links=0 \
        --devices="0,1" --queue_size=2 \
        --backend=serial --skip_existing=0 \
        --run=0
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
from watch.utils.util_param_grid import expand_param_grid
from watch.mlops.old_pipeline_nodes import resolve_pipeline_row
from watch.mlops.old_pipeline_nodes import resolve_package_paths  # NOQA
from watch.mlops.old_pipeline_nodes import Pipeline
from watch.mlops.old_pipeline_nodes import Step
from watch.mlops.old_pipeline_nodes import submit_old_pipeline_jobs


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
        'queue_name': scfg.Value('schedule-eval', help='Name of the queue'),

        'pred_workers': scfg.Value(4, help='number of prediction workers in each process'),

        'shuffle_jobs': scfg.Value(True, help='if True, shuffles the jobs so they are submitted in a random order'),
        'annotations_dpath': scfg.Value(None, help='path to IARPA annotations dpath for IARPA eval'),

        'expt_dvc_dpath': None,
        'data_dvc_dpath': None,

        'check_other_sessions': scfg.Value('auto', help='if True, will ask to kill other sessions that might exist'),
        'queue_size': scfg.Value('auto', help='if auto, defaults to number of GPUs'),

        'out_dpath': scfg.Value('auto', help='The location where predictions / evals will be stored. If "auto", uses teh expt_dvc_dpath'),

        # These enabled flags should probably be pushed off to params
        'enable_pred_trk_pxl': scfg.Value(True, isflag=True, help='BAS heatmap'),
        'enable_pred_trk_poly': scfg.Value(True, isflag=True, help='BAS tracking'),
        'enable_crop': scfg.Value(True, isflag=True, help='SC tracking'),
        'enable_pred_act_pxl': scfg.Value(True, isflag=True, help='SC heatmaps'),
        'enable_pred_act_poly': scfg.Value(True, isflag=True, help='SC tracking'),
        'enable_viz_pred_act_poly': scfg.Value(False, isflag=True, help='if true draw predicted tracks for SC'),

        'enable_eval_trk_pxl': scfg.Value(True, isflag=True, help='BAS heatmap evaluation'),
        'enable_eval_trk_poly': scfg.Value(True, isflag=True, help='BAS tracking evaluation'),
        'enable_eval_act_pxl': scfg.Value(True, isflag=True, help='SC heatmaps evaluation'),
        'enable_eval_act_poly': scfg.Value(True, isflag=True, help='SC tracking evaluation'),
        'enable_viz_pred_trk_poly': scfg.Value(False, isflag=True, help='if true draw predicted tracks for BAS'),
        'enable_links': scfg.Value(True, isflag=True, help='if true enable symlink jobs'),

        'dynamic_skips': scfg.Value(True, isflag=True, help='if true, each a test is appened to each job to skip itself if its output exists'),

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
                trk.pxl.package_fpath:
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
                    - ~/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=73-step=37888.pt.pt
                trk.pxl.data.test_dataset:
                    - ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_bas.json
                trk.pxl.data.tta_time: 0
                trk.pxl.data.chip_overlap: 0.3
                trk.pxl.data.window_space_scale: 10GSD
                trk.pxl.data.input_space_scale: 15GSD
                trk.pxl.data.output_space_scale: 15GSD
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
                ###
                ### SC Pixel Prediction
                ###
                crop.src: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc.json
                crop.regions:
                    - trk.poly.output
                    - truth
                act.pxl.data.test_dataset:
                    - crop.dst
                    # - ~/data/dvc-repos/smart_data_dvc/tmp/KR_R001_0.1BASThresh_40cloudcover_debug10_kwcoco/cropped_kwcoco_for_sc.json
                act.pxl.package_fpath:
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

    out_dpath = config['out_dpath']
    state = ExperimentState(expt_dvc_dpath, '*', storage_dpath=out_dpath)

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
        'trk.pxl.package_fpath': None,
        'trk.pxl.data.test_dataset': None,
        'trk.poly.thresh': 0.1,

        'crop.src': None,
        'crop.context_factor': 1.5,
        'crop.regions': 'truth',

        'act.pxl.package_fpath': None,
        'act.pxl.data.test_dataset': None,
        'act.poly.thresh': 0.1,
    })

    MLOPS_VERSION = 2

    resolved_rows = []
    # Resolve parameters for each row
    for item in ub.ProgIter(all_param_grid, desc='resolving row'):

        if MLOPS_VERSION == 2:
            print('item = {}'.format(ub.repr2(item, nl=1)))
            row = resolve_pipeline_row(grid_item_defaults, state,
                                       region_model_dpath, expt_dvc_dpath, item)
            resolved_rows.append(row)
        elif MLOPS_VERSION == 3:
            ...

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

    environ = {}
    queue = cmd_queue.Queue.create(
        config['backend'], name=config['queue_name'],
        size=queue_size, environ=environ,
        dpath=queue_dpath, gres=GPUS
    )

    virtualenv_cmd = config['virtualenv_cmd']
    if virtualenv_cmd:
        queue.add_header_command(virtualenv_cmd)

    if MLOPS_VERSION == 2:
        submit_old_pipeline_jobs(resolved_rows, queue, config, annotations_dpath)
    elif MLOPS_VERSION == 3:
        ...

    print('queue = {!r}'.format(queue))
    # print(f'{len(queue)=}')
    with_status = 0
    with_rich = 0
    queue.write_network_text()
    queue.rprint(with_status=with_status, with_rich=with_rich)

    for job in queue.jobs:
        # TODO: should be able to set this as a queue param.
        job.log = False

    # RUN
    if config['run']:
        # ub.cmd('bash ' + str(driver_fpath), verbose=3, check=True)
        queue.run(
            block=True,
            # not in cmd_queue 0.1.2?
            # check_other_sessions=config['check_other_sessions']
            with_textual=False,  # needed for backend=tmux
        )
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


__notes__ = """
If this is going to be a real mlops framework, then we need to abstract the
pipeline. The user needs to define what the steps are, but then they need to
explicitly connect them. We can't make the assumptions we are currently using.

Ignore:

    # We can use our CLIs as definitions of the pipeline as long as the config
    # object has enough metadata. With scriptconfig+jsonargparse we should be
    # able to do this.

    import watch.cli.run_metrics_framework
    import watch.cli.run_tracker
    import watch.tasks.fusion.predict

    watch.cli.run_tracker.__config__.__default__

    list(watch.tasks.fusion.predict.make_predict_config().__dict__.keys())


    from watch.tasks.tracking.from_heatmap import NewTrackFunction
    from watch.tasks.tracking.from_heatmap import TimeAggregatedBAS
    from watch.tasks.tracking.from_heatmap import TimeAggregatedSC
    # from watch.tasks.tracking.from_heatmap import TimeAggregatedHybrid

    import jsonargparse
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(TimeAggregatedBAS, nested_key='trk.poly')
    parser.add_class_arguments(TimeAggregatedSC, nested_key='act.poly')
    # parser.add_subclass_arguments(NewTrackFunction, nested_key='poly')
    parser.print_help()

    parser.parse_known_args([])


    import jsonargparse
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--foo')
    args = parser.parse_args(args=[])
    parser.save(args)

"""


if __name__ == '__main__':
    schedule_evaluation(cmdline=True)

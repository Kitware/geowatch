r"""
Helper for scheduling a set of prediction + evaluation jobs

TODO:
    - [ ] Differentiate between pixel models for different tasks.
    - [ ] Allow the output of tracking to feed into activity classification



Example:

    # Dummy inputs, just for demonstration

    python -m watch.mlops.schedule_evaluation \
        --params="
            matrix:
                bas_pxl.package_fpath:
                    - ./my_bas_model1.pt
                    - ./my_bas_model2.pt
                bas_pxl.test_dataset:
                    - ./my_test_dataset/bas_ready_data.kwcoco.json
                bas_pxl.window_space_scale: 15GSD
                bas_pxl.time_sampling:
                    - "auto"
                bas_pxl.input_space_scale:
                    - "15GSD"
                bas_poly.moving_window_size:
                bas_poly.thresh:
                    - 0.1
                    - 0.1
                    - 0.2
                sc_pxl.test_dataset:
                    - crop.dst
                sc_pxl.window_space_scale:
                    - auto
                sc_poly.thresh:
                    - 0.1
                sc_poly.use_viterbi:
                    - 0
                sc_pxl.package_fpath:
                    - my_sc_model1.pt
                    - my_sc_model2.pt
                sc_poly_viz.enabled:
                    - false
        " \
        --expt_dvc_dpath=./my_expt_dir \
        --data_dvc_dpath=./my_data_dir \
        --cache=0 \
        --enable_pred_bas_pxl=1 \
        --enable_pred_bas_poly=1 \
        --enable_eval_bas_pxl=0 \
        --enable_eval_bas_poly=0 \
        --enable_crop=1 \
        --enable_pred_sc_pxl=1 \
        --enable_pred_sc_poly=1 \
        --enable_eval_sc_pxl=0 \
        --enable_eval_sc_poly=0 \
        --enable_viz_pred_bas_poly=0 \
        --enable_viz_pred_sc_poly=0 \
        --enable_links=0 \
        --devices="0,1" --queue_size=2 \
        --backend=serial --skip_existing=0 \
        --run=0

Example:

    # Real data
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

    python -m watch.mlops.schedule_evaluation \
        --params="
            matrix:
                bas_pxl.package_fpath:
                    # - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                    - $DVC_EXPT_DPATH/package_epoch10_step200000.pt
                bas_pxl.test_dataset:
                    - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                    - $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                bas_pxl.window_space_scale:
                    - auto
                    # - "15GSD"
                    # - "30GSD"
                bas_pxl.chip_dims:
                    - "256,256"
                bas_pxl.time_sampling:
                    - "auto"
                bas_pxl.input_space_scale:
                    - "window"
                bas_poly.moving_window_size:
                    - null
                    # - 100
                    # - 200
                bas_poly.thresh:
                    - 0.1
                    # - 0.13
                    # - 0.2
                sc_pxl.window_space_scale:
                    - auto
                sc_pxl.input_space_scale:
                    - "window"
                sc_pxl.chip_dims:
                    - "256,256"
                sc_poly.thresh:
                    - 0.1
                sc_poly.use_viterbi:
                    - 0
                sc_pxl.package_fpath:
                    - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
                bas_poly_eval.enabled: 0
                bas_pxl_eval.enabled: 0
                bas_poly_viz.enabled: 0
                sc_poly_eval.enabled: 0
                sc_pxl_eval.enabled: 0
                sc_poly_viz.enabled: 1
        " \
        --root_dpath=$DVC_EXPT_DPATH/_testpipe \
        --enable_links=1 \
        --devices="0,1" --queue_size=2 \
        --backend=tmux \
        --pipeline=bas \
        --cache=1 --rprint=1 --run=0

        --pipeline=joint_bas_sc


    xdev tree --dirblocklist "_*" my_expt_dir/_testpipe/ --max_files=1
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
# from watch.mlops.old_pipeline_nodes import resolve_pipeline_row
from watch.mlops.old_pipeline_nodes import resolve_package_paths  # NOQA
from watch.mlops.old_pipeline_nodes import Pipeline  # NOQA
from watch.mlops.old_pipeline_nodes import Step  # NOQA
from watch.mlops.old_pipeline_nodes import submit_old_pipeline_jobs  # NOQA


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
        'root_dpath': scfg.Value('auto', help='Where do dump all results'),

        'expt_dvc_dpath': None,
        'data_dvc_dpath': None,

        'pipeline': scfg.Value('joint_bas_sc', help='the name of the pipeline to run'),

        'check_other_sessions': scfg.Value('auto', help='if True, will ask to kill other sessions that might exist'),
        'queue_size': scfg.Value('auto', help='if auto, defaults to number of GPUs'),

        'out_dpath': scfg.Value('auto', help='The location where predictions / evals will be stored. If "auto", uses teh expt_dvc_dpath'),

        # These enabled flags should probably be pushed off to params
        # 'enable_pred_bas_pxl': scfg.Value(True, isflag=True, help='BAS heatmap'),
        # 'enable_pred_bas_poly': scfg.Value(True, isflag=True, help='BAS tracking'),
        # 'enable_crop': scfg.Value(True, isflag=True, help='SC tracking'),
        # 'enable_pred_sc_pxl': scfg.Value(True, isflag=True, help='SC heatmaps'),
        # 'enable_pred_sc_poly': scfg.Value(True, isflag=True, help='SC tracking'),
        # 'enable_viz_pred_sc_poly': scfg.Value(False, isflag=True, help='if true draw predicted tracks for SC'),

        # 'enable_eval_bas_pxl': scfg.Value(True, isflag=True, help='BAS heatmap evaluation'),
        # 'enable_eval_bas_poly': scfg.Value(True, isflag=True, help='BAS tracking evaluation'),
        # 'enable_eval_sc_pxl': scfg.Value(True, isflag=True, help='SC heatmaps evaluation'),
        # 'enable_eval_sc_poly': scfg.Value(True, isflag=True, help='SC tracking evaluation'),
        # 'enable_viz_pred_bas_poly': scfg.Value(False, isflag=True, help='if true draw predicted tracks for BAS'),

        'enable_links': scfg.Value(True, isflag=True, help='if true enable symlink jobs'),
        'cache': scfg.Value(True, isflag=True, help='if true, each a test is appened to each job to skip itself if its output exists'),

        'draw_heatmaps': scfg.Value(1, isflag=True, help='if true draw heatmaps on eval'),
        'draw_curves': scfg.Value(1, isflag=True, help='if true draw curves on eval'),

        'partition': scfg.Value(None, help='specify slurm partition (slurm backend only)'),
        'mem': scfg.Value(None, help='specify slurm memory per task (slurm backend only)'),

        'params': scfg.Value(None, type=str, help='a yaml/json grid/matrix of prediction params'),
        'rprint': scfg.Value(True, isflag=True, help='enable / disable rprint before exec')
    }


def schedule_evaluation(cmdline=False, **kwargs):
    r"""
    First ensure that models have been copied to the DVC repo in the
    appropriate path. (as noted by model_dpath)
    """
    import watch
    from watch.mlops import smart_pipeline
    config = ScheduleEvaluationConfig(cmdline=cmdline, data=kwargs)
    print('ScheduleEvaluationConfig config = {}'.format(ub.repr2(dict(config), nl=1, si=1)))

    if config['root_dpath'] is None:
        expt_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_expt', hardware='auto')
        config['root_dpath'] = expt_dvc_dpath / 'dag_runs'

    root_dpath = ub.Path(config['root_dpath'])

    # Expand paramater search grid
    all_param_grid = expand_param_grid(config['params'])

    # Load the requested pipeline
    dag = smart_pipeline.make_smart_pipeline(config['pipeline'])
    dag.print_graphs()

    # from rich import print
    queue_dpath = root_dpath / '_cmd_queue_schedule'
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

    # Configure a DAG for each row.
    for row_config in ub.ProgIter(all_param_grid, desc='configure dags', verbose=3):
        print('\nrow_config = {}'.format(ub.repr2(row_config, nl=1)))
        dag.configure(
            config=row_config, root_dpath=root_dpath, cache=config['cache'])
        dag.submit_jobs(queue=queue, skip_existing=config['skip_existing'],
                        enable_links=config['enable_links'])

    print('queue = {!r}'.format(queue))
    # print(f'{len(queue)=}')
    with_status = 0
    with_rich = 0
    queue.write_network_text()
    if config['rprint']:
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
    parser.add_class_arguments(TimeAggregatedBAS, nested_key='bas_poly')
    parser.add_class_arguments(TimeAggregatedSC, nested_key='sc_poly')
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

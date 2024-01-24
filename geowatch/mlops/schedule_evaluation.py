r"""
Helper for scheduling a set of prediction + evaluation jobs.

This is the main entrypoint for running a bunch of evaluation jobs over a grid
of parameters. We currently expect that pipelines are predefined in
smart_pipeline.py but in the future they will likely be an external resource
file.

TODO:
    - [ ] Differentiate between pixel models for different tasks.
    - [ ] Allow the output of tracking to feed into activity classification


Example:

    # Dummy inputs, just for demonstration

    python -m geowatch.mlops.schedule_evaluation \
        --params="
            matrix:
                bas_pxl.package_fpath:
                    - ./my_bas_model1.pt
                    - ./my_bas_model2.pt
                bas_pxl.test_dataset:
                    - ./my_test_dataset/bas_ready_data.kwcoco.json
                bas_pxl.window_space_scale: 15GSD
                bas_pxl.time_sampling:
                    - 'auto'
                bas_pxl.input_space_scale:
                    - '15GSD'
                bas_poly_eval.true_site_dpath: null
                bas_poly_eval.true_region_dpath: null
                bas_poly.moving_window_size:
                bas_poly.thresh:
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
        --root_dpath=./my_dag_runs \
        --devices="0,1" --tmux_workers=2 \
        --backend=serial --skip_existing=0 \
        --pipeline=joint_bas_sc \
        --run=0

    python -m geowatch.mlops.schedule_evaluation \
        --params="
            matrix:
                bas_pxl.package_fpath:
                    - ./my_bas_model1.pt
                    - ./my_bas_model2.pt
                bas_pxl.test_dataset:
                    - ./my_test_dataset/bas_ready_data.kwcoco.json
                bas_pxl.window_space_scale: 15GSD
                bas_pxl.time_sampling:
                    - 'auto'
                bas_pxl.input_space_scale:
                    - '15GSD'
                bas_poly.moving_window_size:
                bas_poly.thresh:
                    - 0.1
                    - 0.2
                bas_pxl.enabled: 1
                bas_poly_eval.true_site_dpath: true-site
                bas_poly_eval.true_region_dpath: true-region
        " \
        --root_dpath=./my_dag_runs \
        --devices="0,1" \
        --backend=serial --skip_existing=0 \
        --pipeline=bas \
        --run=0

    # Real inputs, this actually will run something given the DVC repos
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

    SC_MODEL=$DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
    BAS_MODEL=$DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt

    python -m geowatch.mlops.schedule_evaluation \
        --params="
            matrix:
                bas_pxl.package_fpath:
                    - $BAS_MODEL
                bas_pxl.test_dataset:
                    - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                bas_pxl.window_space_scale: 15GSD
                bas_pxl.time_sampling:
                    - "auto"
                bas_pxl.input_space_scale:
                    - "15GSD"
                bas_poly.moving_window_size:
                bas_poly.thresh:
                    - 0.1
                sc_pxl.test_dataset:
                    - crop.dst
                sc_pxl.window_space_scale:
                    - auto
                sc_poly.thresh:
                    - 0.1
                sc_poly.use_viterbi:
                    - 0
                sc_pxl.package_fpath:
                    - $SC_MODEL
                sc_poly_viz.enabled:
                    - false
        " \
        --root_dpath=./my_dag_runs \
        --devices="0,1" --queue_size=2 \
        --backend=serial --skip_existing=0 \
        --pipeline=joint_bas_sc_nocrop \
        --run=0

Example:

    # Real data
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

    python -m geowatch.mlops.schedule_evaluation \
        --params="
            matrix:
                bas_pxl.package_fpath:
                    # - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                    - $DVC_EXPT_DPATH/package_epoch10_step200000.pt
                bas_pxl.test_dataset:
                    - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                    # - $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                bas_pxl.window_space_scale:
                    - auto
                    # - "15GSD"
                    # - "30GSD"
                # bas_pxl.chip_dims:
                #     - "256,256"
                bas_pxl.time_sampling:
                    - "auto"
                # bas_pxl.input_space_scale:
                #     - "window"
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
                bas_poly_eval.enabled: 1
                bas_pxl_eval.enabled: 1
                bas_poly_viz.enabled: 1
                sc_poly_eval.enabled: 1
                sc_pxl_eval.enabled: 1
                sc_poly_viz.enabled: 1
        " \
        --root_dpath=$DVC_EXPT_DPATH/_testpipe \
        --enable_links=1 \
        --devices="0,1" --queue_size=2 \
        --backend=serial \
        --pipeline=bas \
        --cache=1 --rprint=1 --run=1

        --pipeline=joint_bas_sc


    xdev tree --dirblocklist "_*" my_expt_dir/_testpipe/ --max_files=1
"""
import ubelt as ub
import scriptconfig as scfg
from cmd_queue.cli_boilerplate import CMDQueueConfig


class ScheduleEvaluationConfig(CMDQueueConfig):
    """
    Driver for GEOWATCH mlops evaluation scheduling

    Builds commands and optionally executes them via slurm, tmux, or serial
    (i.e. one at a time). This is a [link=https://gitlab.kitware.com/computer-vision/cmd_queue]cmd_queue[/link] CLI.
    """
    __command__ = 'schedule'
    __alias__ = ['mlops_schedule']

    params = scfg.Value(None, type=str, help='a yaml/json grid/matrix of prediction params')

    devices = scfg.Value(None, help=(
        'if using tmux or serial, indicate which gpus are available for use '
        'as a comma separated list: e.g. 0,1'))

    skip_existing = scfg.Value(False, help=(
        'if True dont submit commands where the expected '
        'products already exist'))

    pred_workers = scfg.Value(4, help='number of prediction workers in each process')

    root_dpath = scfg.Value('auto', help=(
        'Where do dump all results. If "auto", uses <expt_dvc_dpath>/dag_runs'))
    pipeline = scfg.Value('joint_bas_sc', help='the name of the pipeline to run. Can also specify this in the params.')

    enable_links = scfg.Value(True, isflag=True, help='if true enable symlink jobs')
    cache = scfg.Value(True, isflag=True, help=(
        'if true, each a test is appened to each job to skip itself if its output exists'))

    draw_heatmaps = scfg.Value(1, isflag=True, help='if true draw heatmaps on pixel eval')
    draw_curves = scfg.Value(1, isflag=True, help='if true draw curves on pixel eval')

    max_configs = scfg.Value(None, help='if specified only run at most this many of the grid search configs')

    queue_size = scfg.Value(None, help='if auto, defaults to number of GPUs')

    print_varied = scfg.Value('auto', isflag=True, help='print the varied parameters')

    def __post_init__(self):
        super().__post_init__()
        if self.queue_name is None:
            self.queue_name = 'schedule-eval'
        if self.queue_size is not None:
            raise Exception('The queue_size argument to schedule evaluation has been removed. Use the tmux_workers argument instead')
            # self.tmux_workers = self.queue_size
        from cmd_queue.util.util_yaml import Yaml
        self.slurm_options = Yaml.coerce(self.slurm_options) or {}

        devices = self.devices
        if devices == 'auto':
            GPUS = _auto_gpus()
        else:
            GPUS = None if devices is None else ensure_iterable(devices)
        self.devices = GPUS


def main(cmdline=True, **kwargs):
    config = ScheduleEvaluationConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('ScheduleEvaluationConfig config = {}'.format(ub.urepr(config, nl=1, sv=1)))
    schedule_evaluation(config)


def schedule_evaluation(config):
    r"""
    First ensure that models have been copied to the DVC repo in the
    appropriate path. (as noted by model_dpath)
    """
    import json
    import pandas as pd
    import rich
    from kwutil import slugify_ext
    from kwutil import util_progress
    from kwutil.util_yaml import Yaml
    from geowatch.mlops import smart_pipeline
    from geowatch.utils.result_analysis import varied_values
    from geowatch.utils.util_param_grid import expand_param_grid

    # Dont put in post-init because it is called by the CLI!
    if config['root_dpath'] in {None, 'auto'}:
        import geowatch
        expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        config['root_dpath'] = expt_dvc_dpath / 'dag_runs'

    root_dpath = ub.Path(config['root_dpath'])

    if config['params'] is not None:
        param_arg = Yaml.coerce(config['params']) or {}
        pipeline = param_arg.pop('pipeline', config.pipeline)

        # Associate BAS datasets and HighRes datasets
        # Convinience to make it less tedious to specify datasets.
        # Hard codes the DVC pattern to associate lowres and hires data.
        # This is not a robust mechanism.
        smart_highres_bundle = param_arg.pop('smart_highres_bundle', None)
        if smart_highres_bundle is not None:
            smart_highres_bundle = ub.Path(smart_highres_bundle)
            assert smart_highres_bundle.exists()
            if 'submatrices' in param_arg:
                raise Exception('cant hack submatrices with submatrices on')

            submatrices = []
            from geowatch import heuristics
            for bas_fpath in param_arg['matrix']['bas_pxl.test_dataset']:
                region_id = heuristics.extract_region_id(ub.Path(bas_fpath).name)
                region_dpath = (smart_highres_bundle / region_id)
                hires_coco_candidates = [
                    region_dpath / f'imgonly-{region_id}.kwcoco.zip',
                    region_dpath / f'imgonly-{region_id}-rawbands.kwcoco.zip',
                ]
                hires_coco_fpath = None
                for cand_fpath in hires_coco_candidates:
                    if cand_fpath.exists():
                        hires_coco_fpath = cand_fpath
                        break
                if hires_coco_fpath is None:
                    raise Exception(f'Expected hires path, but no candidates exist: {hires_coco_candidates}')

                submatrices.append({
                    'bas_pxl.test_dataset': bas_fpath,
                    'sv_crop.crop_src_fpath': hires_coco_fpath,
                    'sc_crop.crop_src_fpath': hires_coco_fpath,
                })
            param_arg['submatrices'] = submatrices

    # Load the requested pipeline
    dag = smart_pipeline.make_smart_pipeline(pipeline)
    dag.print_graphs(smart_colors=1)
    dag.inspect_configurables()

    if config.run:
        mlops_meta = (root_dpath / '_mlops_schedule').ensuredir()
        (root_dpath / '_cmd_queue_schedule').ensuredir()
        # Write some metadata to help aggregate set its defaults automatically
        most_recent_fpath = mlops_meta / 'most_recent_run.json'
        data = {
            'pipeline': str(pipeline),
        }
        most_recent_fpath.write_text(json.dumps(data, indent='    '))

    queue = config.create_queue(gpus=config.devices)

    # Expand paramater search grid
    if config['params'] is not None:
        # print('param_arg = {}'.format(ub.urepr(param_arg, nl=1)))
        all_param_grid = list(expand_param_grid(
            param_arg,
            max_configs=config['max_configs'],
        ))
    else:
        all_param_grid = []

    if len(all_param_grid) == 0:
        print('WARNING: PARAM GRID IS EMPTY')

    # Configure a DAG for each row.
    pman = util_progress.ProgressManager()
    configured_stats = []
    with pman:
        for row_config in pman.progiter(all_param_grid, desc='configure dags', verbose=3):
            dag.configure(
                config=row_config,
                root_dpath=root_dpath,
                cache=config['cache'])
            summary = dag.submit_jobs(
                queue=queue,
                skip_existing=config['skip_existing'],
                enable_links=config['enable_links'])
            configured_stats.append(summary)

    print(f'len(queue)={len(queue)}')

    print_thresh = 30
    if config['print_varied'] == 'auto':
        if len(queue) < print_thresh:
            config['print_varied'] = 1
        else:
            print(f'More than {print_thresh} jobs, skip print_varied. '
                  'If you want to see them explicitly specify print_varied=1')
            config['print_varied'] = 0

    if config['print_varied']:
        # Print config info
        longparams = pd.DataFrame(all_param_grid)
        varied = varied_values(longparams, min_variations=2, dropna=False)
        relevant = longparams[longparams.columns.intersection(varied)]

        def pandas_preformat(item):
            if isinstance(item, str):
                return slugify_ext.smart_truncate(item, max_length=16, trunc_loc=0)
            else:
                return item
        displayable = relevant.applymap(pandas_preformat)
        rich.print(displayable.to_string())

    for job in queue.jobs:
        # TODO: should be able to set this as a queue param.
        job.log = False

    if config.run:
        ub.ensuredir(dag.root_dpath)

    print_kwargs = {
        'with_status': 0,
        'style': "colors",
        'with_locks': 0,
        'exclude_tags': ['boilerplate'],
    }

    rich.print(f'\n\ndag.root_dpath: [link={dag.root_dpath}]{dag.root_dpath}[/link]')
    config.run_queue(queue, print_kwargs=print_kwargs)

    if not config.run:
        driver_fpath = queue.write()
        print('Wrote script: to run execute:\n{}'.format(driver_fpath))

    return dag, queue


def ensure_iterable(inputs):
    return inputs if ub.iterable(inputs) else [inputs]


def _auto_gpus():
    from geowatch.utils.util_nvidia import nvidia_smi
    # TODO: liberate the needed code from netharn
    # Use all unused devices
    GPUS = []
    gpu_info = nvidia_smi()
    for gpu_idx, gpu_info in gpu_info.items():
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

    import geowatch.cli.run_metrics_framework
    import geowatch.cli.run_tracker
    import geowatch.tasks.fusion.predict

    geowatch.cli.run_tracker.__config__.__default__

    list(geowatch.tasks.fusion.predict.make_predict_config().__dict__.keys())

    from geowatch.tasks.tracking.from_heatmap import TimeAggregatedBAS
    from geowatch.tasks.tracking.from_heatmap import TimeAggregatedSC
    # from geowatch.tasks.tracking.from_heatmap import TimeAggregatedHybrid

    import jsonargparse
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(TimeAggregatedBAS, nested_key='bas_poly')
    parser.add_class_arguments(TimeAggregatedSC, nested_key='sc_poly')
    parser.print_help()

    parser.parse_known_args([])


    import jsonargparse
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--foo')
    args = parser.parse_args(args=[])
    parser.save(args)

"""


__config__ = ScheduleEvaluationConfig
__config__.main = main


if __name__ == '__main__':
    main()

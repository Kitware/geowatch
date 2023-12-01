#!/usr/bin/env python3
"""
Script for running the iarpa metrics code with a bit of post-processing.

TODO:
    - [ ] Rename to run_polygon_evaluation.py? Or run_iarpa_metrics.py?

Note currently depends on :py:mod:`iarpa_smart_metrics`, which can be
obtained from:

    https://smartgitlab.com/TE/metrics-and-test-framework
    or
    https://gitlab.kitware.com/smart/metrics-and-test-framework


Example:
    >>> # xdoctest: +REQUIRES(module:iarpa_smart_metrics)
    >>> from geowatch.cli.run_metrics_framework import *  # NOQA
    >>> from geowatch.demo.metrics_demo.generate_demodata import generate_demo_metrics_framework_data
    >>> cmdline = 0
    >>> base_dpath = ub.Path.appdir('geowatch', 'tests', 'test-iarpa-metrics2')
    >>> data_dpath = base_dpath / 'inputs'
    >>> dpath = base_dpath / 'outputs'
    >>> demo_info1 = generate_demo_metrics_framework_data(
    >>>     roi='DR_R001',
    >>>     num_sites=5, num_observations=10, noise=2, p_observe=0.5,
    >>>     p_transition=0.3, drop_noise=0.5, drop_limit=0.5)
    >>> demo_info2 = generate_demo_metrics_framework_data(
    >>>     roi='DR_R002',
    >>>     num_sites=7, num_observations=10, noise=1, p_observe=0.5,
    >>>     p_transition=0.1, drop_noise=0.8, drop_limit=0.5)
    >>> demo_info3 = generate_demo_metrics_framework_data(
    >>>     roi='DR_R003',
    >>>     num_sites=11, num_observations=10, noise=3, p_observe=0.5,
    >>>     p_transition=0.2, drop_noise=0.3, drop_limit=0.5)
    >>> print('demo_info1 = {}'.format(ub.urepr(demo_info1, nl=1)))
    >>> print('demo_info2 = {}'.format(ub.urepr(demo_info2, nl=1)))
    >>> print('demo_info3 = {}'.format(ub.urepr(demo_info3, nl=1)))
    >>> out_dpath = dpath / 'region_metrics'
    >>> merge_fpath = dpath / 'merged.json'
    >>> out_dpath.delete()
    >>> kwargs = {
    >>>     'pred_sites': demo_info1['pred_site_dpath'],
    >>>     'true_region_dpath': demo_info1['true_region_dpath'],
    >>>     'true_site_dpath': demo_info1['true_site_dpath'],
    >>>     'merge': True,
    >>>     'merge_fpath': merge_fpath,
    >>>     'out_dir': out_dpath,
    >>> }
    >>> main(cmdline=False, **kwargs)
    >>> # TODO: visualize


Example:
    >>> # xdoctest: +REQUIRES(module:iarpa_smart_metrics)
    >>> from geowatch.cli.run_metrics_framework import *  # NOQA
    >>> from geowatch.demo.metrics_demo.generate_demodata import generate_demo_metrics_framework_data
    >>> # Test single region case
    >>> cmdline = 0
    >>> base_dpath = ub.Path.appdir('geowatch', 'tests', 'test-iarpa-metrics5')
    >>> data_dpath = (base_dpath / 'inputs').ensuredir()
    >>> dpath = (base_dpath / 'outputs').ensuredir()
    >>> demo_info1 = generate_demo_metrics_framework_data(
    >>>     roi='DR_R004',
    >>>     num_sites=5, num_observations=10, noise=2, p_observe=0.5,
    >>>     p_transition=0.3, drop_noise=0.5, drop_limit=0.5, outdir=data_dpath)
    >>> print('demo_info1 = {}'.format(ub.urepr(demo_info1, nl=1)))
    >>> out_dpath = dpath / 'poly_eval'
    >>> merge_fpath = dpath / 'poly_eval.json'
    >>> out_dpath.delete()
    >>> kwargs = {
    >>>     'pred_sites': demo_info1['pred_site_dpath'],
    >>>     'true_region_dpath': demo_info1['true_region_dpath'],
    >>>     'true_site_dpath': demo_info1['true_site_dpath'],
    >>>     'merge': True,
    >>>     'merge_fpath': merge_fpath,
    >>>     'out_dir': out_dpath,
    >>> }
    >>> main(cmdline=False, **kwargs)
    >>> # TODO: visualize
"""
import os
import json
import shlex
import ubelt as ub
import scriptconfig as scfg
from packaging import version
import warnings


class MetricsConfig(scfg.DataConfig):
    """
    Score IARPA site model GeoJSON files using IARPA's metrics-and-test-framework
    """
    pred_sites = scfg.Value(None,
                            required=True,
                            nargs='*',
                            help=ub.paragraph('''
        List of paths to predicted v2 site models. Or a path to a single text
        file containing the a list of paths to predicted site models.
        All region_ids from these sites will be scored, and it will be assumed
        that there are no other sites in these regions.
        '''))
    gt_dpath = scfg.Value(None,
                          help=ub.paragraph('''
        Path to a local copy of the ground truth annotations,
        https://smartgitlab.com/TE/annotations.  If None, use geowatch_dvc to
        find $DVC_DATA_DPATH/annotations.
        '''))

    true_site_dpath = scfg.Value(None,
                                 help=ub.paragraph('''
        Directory containing true site models. Defaults to
        gt_dpath / site_models
        '''))

    true_region_dpath = scfg.Value(None,
                                   help=ub.paragraph('''
        Directory containing true region models. Defaults to
        gt_dpath / region_models
        '''))

    out_dir = scfg.Value(None,
                         help=ub.paragraph('''
        Output directory where scores will be written. Each
        region will have. Defaults to ./iarpa-metrics-output/
        '''))
    merge = scfg.Value('overwrite',
                       help=ub.paragraph('''
        Merge BAS and SC metrics from all regions and output to
        {out_dir}/merged/.
        'overwrite' = rerun IARPA metrics,
        'read' = assume they exist on disk,
        (TODO 'write' = rerun IARPA metrics if needed.)
        '''))
    merge_fpath = scfg.Value(None,
                             help=ub.paragraph('''
        Forces the merge summary to be written to a specific
        location.
        '''))
    merge_fbetas = scfg.Value([],
                              help=ub.paragraph('''
        A list of BAS F-scores to compute besides F1.
        '''))
    tmp_dir = scfg.Value(None,
                         help=ub.paragraph('''
        If specified, will write temporary data here instead of
        using a     non-persistent directory
        '''))
    enable_viz = scfg.Value(False,
                            isflag=1,
                            help=ub.paragraph('''
        If true, enables iarpa visualizations
        '''))
    name = scfg.Value('unknown',
                      help=ub.paragraph('''
        Short name for the algorithm used to generate the model.
        UNUSED. TODO: incorporate
        '''))

    enable_sc_viz = scfg.Value(False,
                               isflag=1,
                               help=ub.paragraph('''
        If true, enables our SC visualization
        '''))

    load_workers = scfg.Value(0,
                              help=ub.paragraph('''
        The number of workers used to load site models.
        '''))
    parallel = scfg.Value(False, isflag=True,
                          help=ub.paragraph('''
        Innvocate running IARPA T&E metrics in parallel. Note:
        Only works with IARPA T&E metrics version 1.0.0 or greater.
        '''))

    performer = scfg.Value('kit', help='the performer id')

    tau = scfg.Value(0.2, help='T&E tau association threshold param')
    rho = scfg.Value(0.5, help='T&E rho site detection threshold (min proportion of matching polygons)')


def ensure_thumbnails(image_root, region_id, sites):
    '''
    Symlink and organize images in the format the metrics framework expects

    For the region visualizations:
    > image_list = glob(f"{self.image_path}/
    >   {self.region_model.id.replace('_', '/')}/images/*/*/*.jp2")

    For the site visualizations:
    > image_list = glob(f"{self.image_path}/
    >   {gt_ann_id.replace('_', '/')}/crops/*.tif")

    Which becomes:
    {country_code}/
        {region_num}/
            images/
                */
                    */
                        *.jp2
            {site_num}/
                crops/
                    *.tif

    Args:
        image_root: root directory to save under
        region_id: ex. 'KR_R001'
        sites: proposed sites with image paths in the 'source' field
            TODO change to 'misc_info' field
    '''
    image_root = ub.Path(image_root)

    verbose = 0

    # gather images and dates
    site_img_date_dct = dict()
    for site in sites:

        img_date_dct = dict()

        for feat in site['features']:
            props = feat['properties']

            if props['type'] == 'observation':

                img_path = ub.Path(props['source'])
                if img_path.is_file():
                    img_date_dct[img_path] = props['observation_date']
                else:
                    if verbose:
                        print(f'warning: image {img_path}' ' is not a valid path')

            elif props['type'] == 'site':
                site_id = props['site_id']

            else:
                raise ValueError(props['type'])

        site_img_date_dct[site_id] = img_date_dct

    # build region viz
    region_root = image_root.joinpath(*region_id.split('_')) / 'images' / 'a' / 'b'
    region_root.mkdir(parents=True, exist_ok=True)
    for img_path, img_date in ub.dict_union(
            *site_img_date_dct.values()).items():
        link_path = (region_root / '_'.join(
            (img_date.replace('-', ''), img_path.with_suffix('.jp2').name)))
        if img_path.exists():
            ub.symlink(img_path, link_path, verbose=0)
        else:
            if verbose:
                print(f'warning: {img_path=} not found')

    # build site viz
    for site_id, img_date_dct in site_img_date_dct.items():
        site_root = image_root.joinpath(*site_id.split('_')) / 'crops'
        site_root.mkdir(parents=True, exist_ok=True)
        for img_path, img_date in img_date_dct.items():
            # TODO crop
            link_path = (site_root / '_'.join(
                (img_date.replace('-', ''), img_path.with_suffix('.tif').name)))
            if img_path.exists():
                ub.symlink(img_path, link_path, verbose=0)
            else:
                if verbose:
                    print(f'warning: {img_path=} not found')


def main(cmdline=True, **kwargs):
    """
    CommandLine:
        xdoctest -m geowatch.cli.run_metrics_framework main
    """
    from geowatch.utils import util_gis
    from kwcoco.util import util_json
    from geowatch.utils import process_context
    from tempfile import TemporaryDirectory
    import safer
    config = MetricsConfig.cli(cmdline=cmdline, data=kwargs)
    args = config

    config_dict = config.asdict()
    print('config = {}'.format(ub.urepr(config_dict, nl=2, sort=0)))

    try:
        # Do we have the latest and greatest?
        import iarpa_smart_metrics
        IARPA_METRICS_VERSION = version.Version(iarpa_smart_metrics.__version__)
    except Exception:
        raise AssertionError('The iarpa_smart_metrics package should be pip installed ' 'in your virtualenv')
    assert IARPA_METRICS_VERSION >= version.Version('2.5.0')

    # Record the git hash of the metrics code if possible.
    try:
        metrics_modpath = ub.Path(iarpa_smart_metrics.__file__).parent
        gitout = ub.cmd('git rev-parse --short HEAD', cwd=metrics_modpath, check=None)
        IARPA_METRICS_GIT_HASH = gitout.stdout.strip()
    except Exception:
        IARPA_METRICS_GIT_HASH = None

    # Record information about this process
    info = []

    # Args will be serialized in kwcoco, so make sure it can be coerced to json
    jsonified_config = util_json.ensure_json_serializable(config_dict)
    walker = ub.IndexableWalker(jsonified_config)
    for problem in util_json.find_json_unserializable(jsonified_config):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    proc_context = process_context.ProcessContext(
        type='process',
        name='geowatch.cli.run_metrics_framework',
        config=jsonified_config,
        extra={'iarpa_smart_metrics_version': iarpa_smart_metrics.__version__},
    )
    proc_context.start()

    # load pred_sites
    load_workers = config['load_workers']
    pred_site_infos = util_gis.coerce_geojson_paths(config['pred_sites'], return_manifests=True)

    if len(pred_site_infos['manifest_fpaths']) > 1:
        raise Exception('Only expected at most one manifest')

    parent_info = []
    for manifest_fpath in pred_site_infos['manifest_fpaths']:
        # The manifest contains info about how these predictions were computed
        # Grab that if possible.
        print('Load parent info from manifest')
        with open(manifest_fpath, 'r') as file:
            manifest = json.load(file)
        assert (isinstance(manifest, dict) and manifest.get('type', None) == 'tracking_result')
        # The input was a track result json which contains pointers to
        # the actual sites
        parent_info.extend(manifest.get('info', []))

    pred_sites = [
        info['data'] for info in util_gis.coerce_geojson_datas(
            pred_site_infos['geojson_fpaths'], format='json', workers=load_workers)
    ]
    if len(pred_sites) == 0:
        # FIXME: when the tracker produces no results, we fail to score here.
        # Is there a way to produce a valid empty file in the tracker?
        raise Exception('No input predicted sites were given')

    # name = args.name
    true_site_dpath = args.true_site_dpath
    true_region_dpath = args.true_region_dpath

    if true_region_dpath is None or true_site_dpath is None:
        # normalize paths
        if args.gt_dpath is not None:
            gt_dpath = ub.Path(args.gt_dpath).absolute()
        else:
            import geowatch
            data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data')
            gt_dpath = data_dvc_dpath / 'annotations'
            print(f'gt_dpath unspecified, defaulting to {gt_dpath=}')

        if true_region_dpath is None:
            assert gt_dpath.is_dir(), gt_dpath
            true_region_dpath = gt_dpath / 'region_models'
        if true_site_dpath is None:
            assert gt_dpath.is_dir(), gt_dpath
            true_site_dpath = gt_dpath / 'site_models'

    true_region_dpath = ub.Path(true_region_dpath)
    true_site_dpath = ub.Path(true_site_dpath)

    if args.tmp_dir is not None:
        tmp_dpath = ub.Path(args.tmp_dir)
    else:
        temp_dir = TemporaryDirectory(suffix='iarpa-metrics-tmp')
        tmp_dpath = ub.Path(temp_dir.name)

    # split sites by region
    out_dirs = []
    grouped_sites = ub.group_items(pred_sites, lambda site: site['features'][0]['properties']['region_id'])

    main_out_dir = ub.Path(args.out_dir or './iarpa-metrics-output')
    main_out_dir.ensuredir()

    # First build up all of the commands and prepare necessary data for them.
    commands = []

    for region_id, region_sites in ub.ProgIter(sorted(grouped_sites.items()), desc='prepare regions for eval'):

        roi = region_id
        gt_dir = os.fspath(true_site_dpath)

        # Test to see if GT regions exist as they would be checked for in the
        # iarpa_smart_metrics tool.
        from iarpa_smart_metrics.commons import as_local_path
        gt_dir = as_local_path(gt_dir, "annotations/truth/", reg_exp=f".*{roi}.*.geojson")
        gt_dir = ub.Path(gt_dir)
        gt_files = list(gt_dir.glob(f"*{roi}*.geojson"))

        if len(gt_files) == 0:
            warnings.warn(f'No truth for region: {roi}. Skipping')
            continue

        site_dpath = (tmp_dpath / 'site' / region_id).ensuredir()
        image_dpath = (tmp_dpath / 'image').ensuredir()

        out_dir = (main_out_dir / region_id).ensuredir()
        out_dirs.append(out_dir)

        # doctor site_dpath for expected structure
        pred_site_sub_dpath = site_dpath / 'latest' / region_id
        pred_site_sub_dpath.ensuredir()

        # copy site models to site_dpath
        for site in region_sites:
            geojson_fpath = pred_site_sub_dpath / (site['features'][0]['properties']['site_id'] + '.geojson')
            with safer.open(geojson_fpath, 'w', temp_file=not ub.WIN32) as f:
                json.dump(site, f)

        ensure_thumbnails(image_dpath, region_id, region_sites)

        key_to_disable_flag = {
            'region': '--no-viz-region',  # we often want this enabled
            'slices': '--no-viz-slices',
            'detection_table': '--no-viz-detection-table',
            'comparison_table': '--no-viz-comparison-table',
            'associate_metrics': '--no-viz-associate-metrics',
            'activity_metrics': '--no-viz-activity-metrics',
        }

        if isinstance(args.enable_viz, str):
            # Allow the user to enable specific visualizations
            chosen = set(args.enable_viz.split(','))
        elif args.enable_viz:
            # Enable all visualizations (usually a bad idea)
            # warnings.warn(ub.paragraph(
            #     '''
            #     All IARPA visualizations were enabled.  Try setting
            #     --enable_viz=region to get only the useful visualizations
            #     '''))
            warnings.warn(ub.paragraph(
                '''
                When enable_viz is True, we default to only using "region"
                Can also set things like --enable_viz="region,slices,detection_table"
                etc...
                '''))
            # chosen = set(key_to_disable_flag)
            chosen = {'region'}
        else:
            chosen = set()

        to_disable = set(key_to_disable_flag) - chosen
        to_enable = set(key_to_disable_flag) & chosen

        viz_flags = [key_to_disable_flag[k] for k in sorted(to_disable)]
        viz_flags += [key_to_disable_flag[k].replace('--no-', '--')
                      for k in sorted(to_enable)]

        run_eval_command = [
            'python',
            '-m',
            'iarpa_smart_metrics.run_evaluation',
            '--roi',
            roi,
            '--gt_dir',
            os.fspath(gt_dir),
            '--rm_dir',
            os.fspath(true_region_dpath),
            '--sm_dir',
            os.fspath(pred_site_sub_dpath),
            '--image_dir',
            os.fspath(image_dpath),
            '--output_dir',
            os.fspath(out_dir),
            ## Restrict to make this faster
            '--tau', str(config.tau),
            '--rho', str(config.rho),
            '--activity', 'overall',
            '--loglevel', 'ERROR',
            f'--performer={config.performer}',
            '--eval_num=0',
            '--eval_run_num=0',
            # '--no-db',
            '--sequestered_id', '',
            # 'seq',  # default None broken on autogen branch
        ]
        # Add parallel flag if requested
        if args.parallel:
            run_eval_command += ['--parallel']
        else:
            run_eval_command += ['--serial']

        run_eval_command += viz_flags
        # run metrics framework
        cmd = shlex.join(run_eval_command)
        region_invocation_text = ub.codeblock('''
            #!/bin/bash
            __doc__="
            This is an auto-generated file that records the command used to
            generate this evaluation of this particular region.
            "
            ''') + chr(10) + cmd + chr(10)
        # Dump this command to disk for reference and debugging.
        (out_dir / 'invocation.sh').write_text(region_invocation_text)
        commands.append(cmd)

    if 0:
        import cmd_queue
        queue = cmd_queue.Queue.create(backend='serial')
        for cmd in commands:
            queue.submit(cmd)
            # TODO: make command queue stop on the first failure?
            queue.run()
        # if queue.read_state()['failed']:
        #     raise Exception('jobs failed')
    else:
        import subprocess
        for cmd in commands:
            if args.merge != 'read':
                try:
                    ub.cmd(cmd, verbose=3, check=True, shell=True)
                except subprocess.CalledProcessError:
                    print('error in metrics framework, probably due to zero '
                          'TP site matches or a region without site truth.')

    if len(grouped_sites) == 1:
        print('Dump confusion')
        # In the case where there is one region:
        # Write a the invocation for confusion analysis.
        region_id = list(grouped_sites.keys())[0]
        confusion_analysis_text = ub.codeblock(
            rf'''
            #!/bin/bash
            __doc__="
            This is an auto-generated file that records the command used to
            generate this evaluation of multiple regions.
            "
            python -m geowatch.mlops.confusor_analysis \
                --metrics_node_dpath={main_out_dir} \
                --out_dpath={main_out_dir}/confusion_analysis \
                --true_region_dpath={true_region_dpath} \
                --true_site_dpath={true_site_dpath} \
                --region_id={region_id} \
                --viz_sites=True \
                --reload=0 "$@"
            ''')
        cfsn_invoke_fpath = (main_out_dir / 'confusion_analysis.sh')
        cfsn_invoke_fpath.write_text(confusion_analysis_text)
        import stat
        cfsn_invoke_fpath.chmod(cfsn_invoke_fpath.stat().st_mode | stat.S_IXUSR)

    print('out_dirs = {}'.format(ub.urepr(out_dirs, nl=1)))
    if args.merge and out_dirs:
        from geowatch.tasks.metrics.merge_iarpa_metrics import merge_metrics_results
        from geowatch.tasks.metrics.merge_iarpa_metrics import iarpa_bas_color_legend
        import kwimage

        if args.merge_fpath is None:
            merge_dpath = (main_out_dir / 'merged').ensuredir()
            merge_fpath = merge_dpath / 'summary2.json'
        else:
            merge_fpath = ub.Path(args.merge_fpath)
            merge_dpath = merge_fpath.parent.ensuredir()

        region_dpaths = out_dirs

        context = proc_context.stop()
        context['IARPA_METRICS_VERSION'] = str(IARPA_METRICS_VERSION)
        context['IARPA_METRICS_GIT_HASH'] = IARPA_METRICS_GIT_HASH
        info.append(context)

        json_data, bas_df, sc_df, best_bas_rows = merge_metrics_results(region_dpaths, true_site_dpath,
                                                                        true_region_dpath, args.merge_fbetas)

        # TODO: parent info should probably belong to info itself
        json_data['info'] = info
        json_data['parent_info'] = parent_info

        merge_dpath = ub.Path(merge_dpath).ensuredir()

        with safer.open(merge_fpath, 'w', temp_file=not ub.WIN32) as f:
            json.dump(json_data, f, indent=4)
        print('merge_fpath = {!r}'.format(merge_fpath))

        # Consolodate visualizations
        combined_viz_dpath = (merge_dpath / 'region_viz_overall')

        # Write a legend to go with the BAS viz
        if config.enable_viz:
            combined_viz_dpath.ensuredir()
            legend_img = iarpa_bas_color_legend()
            legend_fpath = (combined_viz_dpath / 'bas_legend.png')
            kwimage.imwrite(legend_fpath, legend_img)

        bas_df.to_pickle(merge_dpath / 'bas_df.pkl')
        sc_df.to_pickle(merge_dpath / 'sc_df.pkl')

        # Symlink to visualizations
        for dpath in region_dpaths:
            overall_dpath = dpath / 'overall'
            viz_dpath = (overall_dpath / 'bas' / 'region').ensuredir()

        for viz_fpath in viz_dpath.iterdir():
            combined_viz_dpath.ensuredir()
            viz_link = viz_fpath.augment(dpath=combined_viz_dpath)
            ub.symlink(viz_fpath, viz_link, verbose=1)

        # viz SC
        if config.enable_sc_viz:
            combined_viz_dpath.ensuredir()
            from geowatch.tasks.metrics.viz_sc_results import viz_sc
            viz_sc(region_dpaths, true_site_dpath, true_region_dpath, combined_viz_dpath)


__config__ = MetricsConfig
__config__.main = main

if __name__ == '__main__':
    main()

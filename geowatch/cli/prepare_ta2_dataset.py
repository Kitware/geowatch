r"""
Builds a multi-region dataset.

An end-to-end script for calling all the scripts needed to

* Pulls the STAC catalog that points to processed large image tiles
* Creates a virtual Uncropped kwcoco dataset that points to the large image tiles
* Crops the dataset to create an aligned TA2 dataset

See Also:
    ~/code/geowatch/scripts/prepare_drop4.sh
    ~/code/geowatch/scripts/prepare_drop5.sh

CommandLine:

    # Create a demo region file, and create vairables that point at relevant
    # paths, which are by default written in your ~/.cache folder
    xdoctest -m geowatch.demo.demo_region demo_khq_region_fpath
    REGION_FPATH="$HOME/.cache/geowatch/demo/annotations/KHQ_R001.geojson"
    SITE_GLOBSTR="$HOME/.cache/geowatch/demo/annotations/KHQ_R001_sites/*.geojson"

    # The "name" of the new dataset
    DATASET_SUFFIX=Demo-TA2-KHQ

    # Set this to where you want to build the dataset
    DEMO_DPATH=$PWD/prep_ta2_demo

    mkdir -p "$DEMO_DPATH"

    # This is a string code indicating what STAC endpoint we will pull from
    SENSORS="sentinel-2-l2a"

    # Depending on the STAC endpoint, some parameters may need to change:
    # collated - True for IARPA endpoints, Usually False for public data
    # requester_pays - True for public landsat
    # api_key - A secret for non public data

    export SMART_STAC_API_KEY=""
    export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

    # Construct the TA2-ready dataset
    python -m geowatch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --cloud_cover=100 \
        --stac_query_mode=auto \
        --sensors "$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --requester_pays=True \
        --dvc_dpath="$DEMO_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_FPATH" \
        --site_globstr="$SITE_GLOBSTR" \
        --fields_workers=8 \
        --convert_workers=0 \
        --align_workers=26 \
        --cache=1 \
        --skip_existing=0 \
        --ignore_duplicates=1 \
        --target_gsd=30 \
        --visualize=True \
        --max_products_per_region=10 \
        --backend=serial \
        --run=0

    geowatch visualize $HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/data.kwcoco_c9ea8bb9.json

TODO:
    handle GE01 and WV01 platforms

CommandLine:
    xdoctest -m geowatch.cli.prepare_ta2_dataset __doc__:0

Example:
    >>> from geowatch.cli.prepare_ta2_dataset import *  # NOQA
    >>> import ubelt as ub
    >>> dpath = ub.Path.appdir('geowatch/test/prep_ta2_dataset').delete().ensuredir()
    >>> from geowatch.geoannots import geomodels
    >>> # Write dummy regions / sites
    >>> for rng in [0, 1, 3]:
    >>>     region, sites = geomodels.RegionModel.random(rng=rng, with_sites=True)
    >>>     region_dpath = (dpath / 'region_models').ensuredir()
    >>>     site_dpath = (dpath / 'site_models').ensuredir()
    >>>     region_fpath = region_dpath / (region.region_id + '.geojson')
    >>>     region_fpath.write_text(region.dumps())
    >>>     for site in sites:
    >>>         site_fpath = site_dpath / (site.site_id + '.geojson')
    >>>         site_fpath.write_text(site.dumps())
    >>> # Prepare config and test a dry run
    >>> kwargs = PrepareTA2Config()
    >>> kwargs['dataset_suffix'] = 'DEMO_DOCTEST'
    >>> kwargs['run'] = 0
    >>> kwargs['stac_query_mode'] = 'auto'
    >>> kwargs['regions'] = region_dpath
    >>> kwargs['sites'] = site_dpath
    >>> kwargs['backend'] = 'serial'
    >>> kwargs['visualize'] = 1
    >>> kwargs['collated'] = [True]
    >>> kwargs['out_dpath'] = '.'
    >>> cmdline = 0
    >>> PrepareTA2Config.main(cmdline=cmdline, **kwargs)
"""


import scriptconfig as scfg
import ubelt as ub
import warnings
from cmd_queue.cli_boilerplate import CMDQueueConfig


class PrepareTA2Config(CMDQueueConfig):

    queue_name = scfg.Value('prep-ta2-dataset', group='cmd-queue', help='name for the command queue')

    dataset_suffix = scfg.Value(None, help='')

    cloud_cover = scfg.Value(10, help=ub.paragraph(
            '''
            maximum cloud cover percentage (ignored if s3_fpath given)
            '''))

    sensors = scfg.Value('L2', type=str, help='(ignored if s3_fpath given)')

    max_products_per_region = scfg.Value(None, help=ub.paragraph(
            '''
            does uniform affinity sampling over time to filter down to
            this many results per region
            '''))

    stac_query_mode = scfg.Value(None, group='stac', help=ub.paragraph(
            '''
            if set to auto we try to make the .input files. Mutex with
            s3_fpath
            '''))

    api_key = scfg.Value('env:SMART_STAC_API_KEY', group='stac', help=ub.paragraph(
            '''
            The API key or where to get it (ignored if s3_fpath given)
            '''))

    s3_fpath = scfg.Value(None, group='stac', help=ub.paragraph(
            '''
            A list of .input files which were the results of an existing
            stac query. Mutex with stac_query_* args. Mutex with
            sensors.
            '''), nargs='+')

    aws_profile = scfg.Value('iarpa', group='stac', help=ub.paragraph(
            '''
            AWS profile to use for remote data access
            '''))

    out_dpath = scfg.Value('auto', alias=['dvc_dpath'], help=ub.paragraph(
            '''
            This is the path that all resulting files will be written
            to. Defaults the the phase2 DATA_DVC_DPATH
            '''))

    collated = scfg.Value([True], help=ub.paragraph(
            '''
            set to false if the input data is not collated
            '''), nargs='+')

    max_regions = scfg.Value(None, help=None)

    query_workers = scfg.Value(0, help='workers for STAC search')

    convert_workers = scfg.Value(0, help=ub.paragraph(
            '''
            workers for stac-to-kwcoco script. Keep this set to zero!
            '''))

    fields_workers = scfg.Value('min(avail,max(all/2,8))', type=str, help='workers for add-watch-fields script')

    align_workers = scfg.Value(0, group='align', help='primary workers for align script')

    align_aux_workers = scfg.Value(0, group='align', help=ub.paragraph(
            '''
            threads per align process (typically set this to 0)
            '''))

    align_tries = scfg.Value(2, help=ub.paragraph(
            '''
            The maximum number of times to retry failed gdal warp
            commands before stopping.
            '''))

    image_timeout = scfg.Value('8hours', help=ub.paragraph(
            '''
            The maximum amount of time to spend pulling down a all image
            assets before giving up
            '''))

    asset_timeout = scfg.Value('4hours', help=ub.paragraph(
            '''
            The maximum amount of time to spend pulling down a single
            image asset before giving up
            '''))

    ignore_duplicates = scfg.Value(1, help='workers for align script')

    visualize = scfg.Value(0, isflag=1, help='if True runs visualize')

    visualize_only_boxes = scfg.Value(True, isflag=1, help='if False will draw full polygons')

    verbose = scfg.Value(0, help=ub.paragraph(
            '''
            help control verbosity (just align for now)
            '''))

    requester_pays = scfg.Value(0, help=ub.paragraph(
            '''
            if True, turn on requester_pays in ingress. Needed for
            official L1/L2 catalogs.
            '''))

    debug = scfg.Value(False, isflag=1, help=ub.paragraph(
            '''
            if enabled, turns on debug visualizations
            '''))

    select_images = scfg.Value(False, help='if enabled only uses select images')

    include_channels = scfg.Value(None, group='align', help='specific channels to use in align crop')

    exclude_channels = scfg.Value(None, group='align', help=ub.paragraph(
            '''
            specific channels to NOT use in align crop
            '''))

    target_gsd = scfg.Value(10, group='align', help=None)

    force_min_gsd = scfg.Value(None, group='align', help=None)

    align_keep = scfg.Value('img', group='align', help=ub.paragraph(
            '''
            if the coco align script caches or recomputes images / rois
            '''), choices=['img', 'img-roi', 'none', None])

    force_nodata = scfg.Value(None, group='align', help=ub.paragraph(
            '''
            if specified, forces nodata to this value
            '''))
    splits = scfg.Value(False, isflag=1, help='if True do splits')

    regions = scfg.Value('annotations/region_models', alias=['region_globstr'], help=ub.paragraph(
            '''
            region model globstr (relative to the dvc path, unless
            absolute or prefixed by "./")
            '''))

    sites = scfg.Value(None, alias=['site_globstr'], help=ub.paragraph(
            '''
            site model globstr (relative to the dvc path, unless
            absolute or prefixed by "./")
            '''))

    propogate_strategy = scfg.Value('NEW-SMART', help='changes propogation behavior')

    remove_broken = scfg.Value(True, isflag=1, help=ub.paragraph(
            '''
            if True, will remove any image that fails population (e.g.
            caused by a 404)
            '''))

    skip_populate_errors = scfg.Value(False, isflag=1, help=ub.paragraph(
            '''
            if True, skip processing any bands that raise errors (e.g.
            caused by permission errors on S3)
            '''))

    cache = scfg.Value(1, isflag=1, group='queue-related', help=ub.paragraph(
            '''
            If 1 or 0 globally enable/disable caching. If a comma
            separated list of strings, only cache those stages
            '''))

    skip_existing = scfg.Value(False, group='queue-related', help=ub.paragraph(
            '''
            Unlike cache=1, which checks for file existence at runtime,
            this will explicitly not submit any job with a product that
            already exist
            '''))

    rpc_align_method = scfg.Value('orthorectify', help=ub.paragraph(
            '''
            Can be one of: (1) orthorectify - which uses gdalwarp with
            -rpc if available otherwise falls back to affine transform,
            (2) affine_warp - which ignores RPCs and uses the affine
            transform in the geotiff metadata.
            '''))

    sensor_to_time_window = scfg.Value(None, help=ub.paragraph(
        '''
        Specify a yaml mapping from a sensor to a time window. We will chunk up
        candidate images based on this window and only choose 1 image per
        chunk with the lowest cloud cover using earlier images as tiebreakers.
        '''))

    reproject_annotations = scfg.Value(True, isflag=True, help=ub.paragraph(
        '''
        If True and site models are given, project annotations onto the coco
        files after they are cropped / aligned.
        '''))

    final_union = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        If True, union all regions into a single kwcoco file at the end
        to represent the entired dataset.
        '''))

    hack_lazy = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Hack lazy is a proof of concept with the intent on speeding
            up the download / cropping of data by flattening the gdal
            processing into a single queue of parallel processes
            executed via a command queue. By running once with this flag
            on, it will execute the command queue, and then running
            again, it should see all of the data as existing and
            construct the aligned kwcoco dataset as normal.
            '''))

    @classmethod
    def _register_main(cls, func):
        cls.main = func
        return func


__config__ = PrepareTA2Config


def _justkeys(d):
    return set(d.keys())
    # return d


@__config__._register_main
def main(cmdline=False, **kwargs):
    """

    Ignore:
        from geowatch.cli.prepare_ta2_dataset import *  # NOQA
        cmdline = False
        kwargs = {
            'dataset_suffix': 'TA1_FULL_SEQ_KR_S001_CLOUD_LT_10',
            's3_fpath': 's3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working/KR_S001.unique.fixed_ls_ids.cloudcover_lt_10.output',
        }
        kwargs = {
            'dataset_suffix': 'Drop2-2022-02-04',
            's3_fpath': 's3://kitware-smart-watch-data/processed/ta1/drop2_20220204/PE/coreg_and_brdf/watch-coreg-and-brdf_PE.output',
        }

    """
    config = PrepareTA2Config.cli(cmdline=cmdline, data=kwargs, strict=True)
    from kwutil import slugify_ext
    from geowatch.utils import util_gis
    import rich
    from geowatch.mlops.pipeline_nodes import Pipeline
    rich.print('config = {}'.format(ub.urepr(dict(config), nl=1)))

    out_dpath = config['out_dpath']
    if out_dpath == 'auto':
        import geowatch
        out_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    out_dpath = ub.Path(out_dpath)

    aws_profile = config['aws_profile']

    aligned_bundle_name = f'Aligned-{config["dataset_suffix"]}'
    uncropped_bundle_name = f'Uncropped-{config["dataset_suffix"]}'

    if 1:
        uncropped_dpath = out_dpath / uncropped_bundle_name
        aligned_kwcoco_bundle = out_dpath / aligned_bundle_name
    else:
        uncropped_dpath = ub.Path('.') / uncropped_bundle_name
        aligned_kwcoco_bundle = ub.Path('.') / aligned_bundle_name

    uncropped_query_dpath = uncropped_dpath / '_query/items'
    uncropped_ingress_dpath = uncropped_dpath / 'ingress'

    # Ignore these regions (only works in separate region queue mode)
    region_id_blocklist = {
        # 'IN_C000',  # bad files
        # 'ET_C000',  # 404 errors
    }

    # Job environments are specific to single jobs
    job_environs = [
        # 'PROJ_DEBUG=3',
        f'AWS_DEFAULT_PROFILE={aws_profile}',
    ]
    if config['requester_pays']:
        job_environs.append("AWS_REQUEST_PAYER='requester'")
    job_environ_str = ' '.join(job_environs)
    if job_environ_str:
        job_environ_str += ' '

    # Global environs are given to all jobs
    api_key = config['api_key']
    environ = {}
    # https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_DISABLE_READDIR_ON_OPEN
    environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
    if api_key is not None and api_key.startswith('env:'):
        import os
        # NOTE!!!
        # THIS WILL LOG YOUR SECRET KEY IN PLAINTEXT!!!
        # TODO: figure out how to pass the in-environment secret key
        # to the tmux sessions.
        api_key_name = api_key[4:]
        api_key_val = os.environ.get(api_key_name, None)
        if api_key_val is None:
            warnings.warn('The requested API key was not available')
        else:
            environ[api_key_name] = api_key_val

    default_collated = config['collated'][0]

    # The pipeline is a layer on top of cmd-queue that will handle connecting
    # inputs / outputs for us. At the time of writing we are still doing that
    # explicitly, but we could remove that code and just rely on implicit
    # dependencies based in in / out paths.
    new_pipeline = Pipeline()

    pipe_config = {}

    stac_jobs = []
    if config['stac_query_mode'] == 'auto':
        # Each region gets their own job in the queue
        # Note: this requires the annotation files to exist on disk.  or we
        # have to write a mechanism that lets the explicit relative path be
        # specified.

        # Note pattern matching is driven by kwutil.util_path.coerce_patterned_paths
        region_file_fpaths = util_gis.coerce_geojson_paths(config.regions)

        if config.sites:
            region_site_fpaths = util_gis.coerce_geojson_paths(config.sites)
        else:
            region_site_fpaths = []

        # Assign site models to region files
        ASSIGN_BY_FPATH = True
        if ASSIGN_BY_FPATH:
            # This is not robust, but it doesn't require touching the disk
            region_id_to_fpath = {p.stem: p for p in region_file_fpaths}
            site_id_to_fpath = {p.stem: p for p in region_site_fpaths}
            region_id_to_site_fpaths = ub.ddict(list)
            for site_id, site_fpaths in site_id_to_fpath.items():
                region_id, site_num = site_id.rsplit('_', maxsplit=1)
                region_id_to_site_fpaths[region_id].append(site_fpaths)

            if 1:
                regions_without_sites = set(region_id_to_fpath) - set(region_id_to_site_fpaths)
                sites_without_regions = set(region_id_to_site_fpaths) - set(region_id_to_fpath)
                regions_without_sites_str = slugify_ext.smart_truncate(ub.urepr(regions_without_sites, nl=1), max_length=1000, head="~~\n~~\n", tail="\n~~\n~~")
                sites_without_regions_str = slugify_ext.smart_truncate(ub.urepr(sites_without_regions, nl=1), max_length=1000, head="~~\n~~\n", tail="\n~~\n~~")
                print('len(regions_with_sites) = ' + str(len(region_id_to_site_fpaths)))
                print('len(sites_with_region) = ' + str(sum(map(len, region_id_to_site_fpaths.values()))))
                print(f'regions_without_sites={regions_without_sites_str}')
                print(f'sites_without_regions={sites_without_regions_str}')
        else:
            raise NotImplementedError(
                'TODO: implement more robust alternative that reads '
                'file data to make assignment if needed')

        if config['max_regions'] is not None:
            region_file_fpaths = region_file_fpaths[:config['max_regions']]

        region_to_site_pattern = {}
        USE_TNE_NAME_HACK = True
        if USE_TNE_NAME_HACK:
            # fixes a scalability issue, but we require a better site
            # referencing system to make this work efficiently in general.
            # import kwutil
            for region_id, site_fpaths in region_id_to_site_fpaths.items():
                region_fpath = region_id_to_fpath[region_id]
                region_to_site_pattern[region_id] = region_fpath.parent.parent / 'site_models' / region_id + '_*.geojson'

        print(f'region_file_fpaths={slugify_ext.smart_truncate(ub.urepr(sorted(region_file_fpaths), nl=1), max_length=1000)}')
        for region_id, region_fpath in region_id_to_fpath.items():
            if region_id in region_id_blocklist:
                continue

            region_inputs_fpath = (uncropped_query_dpath / (region_id + '.input'))
            final_region_fpath = region_fpath

            stac_search_node = new_pipeline.submit(
                name=f'stac-search-{region_id}',
                perf_params={
                    'api_key': config.api_key,
                    'query_workers': config.query_workers,
                    'verbose': 2,
                },
                algo_params={
                    'search_json': 'auto',
                    'cloud_cover': config.cloud_cover,
                    'sensors': config.sensors,
                    'max_products_per_region': config.max_products_per_region,
                    'append_mode': False,
                    'mode': 'area',
                },
                executable=ub.codeblock(
                    r'''
                    python -m geowatch.cli.stac_search
                    '''),
                in_paths=_justkeys({
                    'region_file': final_region_fpath,
                }),
                out_paths={
                    'outfile': region_inputs_fpath,
                },
                group_dname=uncropped_bundle_name,
                # node_dpath
            )

            # All other paths fall out after specifying this one
            stac_search_node.configure({
                'region_file': final_region_fpath,
            })
            pipe_config[stac_search_node.name + '.region_file'] = final_region_fpath

            # NOTE: The YAML list can get too long pretty quickly.
            # So we can't pass this as site-globstr. However, we could write a
            # file with this info and pass that in...
            # from kwutil.util_yaml import Yaml
            # sites = region_id_to_site_fpaths.get(region_id, None)
            # explicit_sites = Yaml.dumps(list(map(os.fspath, sites)))

            site_globstr = region_to_site_pattern.get(region_id, None)
            if site_globstr is None:
                site_globstr = config.sites

            stac_jobs.append({
                'name': region_id,
                'node': stac_search_node,
                'inputs_fpath': region_inputs_fpath,
                'region_globstr': final_region_fpath,
                # 'site_globstr': explicit_sites,
                'site_globstr': site_globstr,
                'collated': default_collated,
            })
    else:
        # Typically unused
        s3_fpath_list = config['s3_fpath']
        collated_list = config['collated']
        if len(collated_list) != len(s3_fpath_list):
            print('Indicate if each s3 path is collated or not')
        stac_jobs = []
        for s3_fpath, collated in zip(s3_fpath_list, collated_list):
            stac_jobs.append({
                'node': None,
                'name': ub.Path(s3_fpath).stem,
                'inputs_fpath': s3_fpath,
                'region_globstr': config.regions,
                'site_globstr': config.sites,
                'collated': collated,
            })

    uncropped_fielded_jobs = []
    for stac_job in stac_jobs:
        s3_fpath = ub.Path(stac_job['inputs_fpath'])
        s3_name = stac_job['name']
        parent_node = stac_job['node']
        collated = stac_job['collated']
        uncropped_query_fpath = uncropped_query_dpath / s3_fpath.name
        uncropped_catalog_fpath = uncropped_ingress_dpath / f'catalog_{s3_name}.json'

        if not str(s3_fpath).startswith('s3'):
            # Don't really need to copy anything in this case.
            uncropped_query_fpath = s3_fpath
            grab_node = parent_node
        else:
            # GRAB Input STAC List
            grab_node = new_pipeline.submit(
                name=f's3-pull-inputs-{s3_name}',
                executable=ub.codeblock(
                    f'''
                    aws s3 --profile {aws_profile} cp "{s3_fpath}" "{uncropped_query_dpath}"
                    '''),
                in_paths=_justkeys({
                    's3_fpath': s3_fpath,
                }),
                out_paths={
                    'uncropped_query_fpath': uncropped_query_fpath,
                },
                _no_outarg=True,
                _no_inarg=True,
            )
            parent_node.outputs['outfile'].connect(grab_node.inputs['s3_fpath'])
            print('warning... config might be off...')

        ingress_node = new_pipeline.submit(
            name=f'baseline_ingress-{s3_name}',
            perf_params={
                'virtual': True,
                'jobs': 'avail',
                'aws_profile': aws_profile,
                'requester_pays': config.requester_pays,
            },
            executable=ub.codeblock(
                r'''
                python -m geowatch.cli.baseline_framework_ingress
                '''),
            in_paths=_justkeys({
                'input_path': uncropped_query_fpath,
            }),
            out_paths={
                'catalog_fpath': uncropped_catalog_fpath,
            },
            group_dname=uncropped_bundle_name,
        )
        if grab_node is not None:
            try:
                grab_node.outputs['uncropped_query_fpath'].connect(ingress_node.inputs['input_path'])
            except KeyError:
                grab_node.outputs['outfile'].connect(ingress_node.inputs['input_path'])

        uncropped_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}.kwcoco.zip'

        convert_node = new_pipeline.submit(
            name=f'stac_to_kwcoco-{s3_name}',
            algo_params={
                'from_collated': collated,
                'ignore_duplicates': config.ignore_duplicates,
            },
            perf_params={
                'jobs': config.convert_workers,
            },
            executable=ub.codeblock(
                fr'''
                {job_environ_str}python -m geowatch.cli.stac_to_kwcoco
                '''),
            in_paths=_justkeys({
                'input_stac_catalog': uncropped_catalog_fpath,
            }),
            out_paths={
                'outpath': uncropped_kwcoco_fpath,
            },
            group_dname=uncropped_bundle_name,
        )
        ingress_node.outputs['catalog_fpath'].connect(convert_node.inputs['input_stac_catalog'])

        uncropped_fielded_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}_fielded.kwcoco.zip'

        add_fields_node = new_pipeline.submit(
            name=f'coco_add_watch_fields-{s3_name}',
            algo_params={
                'enable_video_stats': False,
                'target_gsd': config.target_gsd,
                'remove_broken': config.remove_broken,
                'skip_populate_errors': config.skip_populate_errors,
            },
            perf_params={
                'overwrite': 'warp',
                'workers': config.fields_workers,
            },
            executable=ub.codeblock(
                fr'''
                {job_environ_str}python -m geowatch.cli.coco_add_watch_fields
                '''),
            in_paths=_justkeys({
                'src': uncropped_kwcoco_fpath,
            }),
            out_paths={
                'dst': uncropped_fielded_kwcoco_fpath,
            },
            group_dname=uncropped_bundle_name,
        )
        convert_node.outputs['outpath'].connect(add_fields_node.inputs['src'])

        uncropped_fielded_jobs.append({
            'name': stac_job['name'],
            'node': add_fields_node,
            'uncropped_fielded_fpath': uncropped_fielded_kwcoco_fpath,
            'region_globstr': stac_job['region_globstr'],
            'site_globstr': stac_job['site_globstr'],
        })

    alignment_input_jobs = []
    # TODO: make use of region_id_to_site_fpaths
    # to build a better site globstr (might need to write a tempoaray file
    # to point to)
    # final_site_globstr = _coerce_globstr(config['sites'])
    for info in uncropped_fielded_jobs:
        toalign_info = info.copy()
        name = toalign_info['name'] = info['name']
        toalign_info['aligned_imgonly_fpath'] = aligned_kwcoco_bundle / name / f'imgonly-{name}-rawbands.kwcoco.zip'
        toalign_info['aligned_imganns_fpath'] = aligned_kwcoco_bundle / name / f'imganns-{name}-rawbands.kwcoco.zip'
        # TODO: take only the corresponding set of site models here.
        toalign_info['site_globstr'] = info['site_globstr']
        toalign_info['region_globstr'] = info['region_globstr']
        alignment_input_jobs.append(toalign_info)

    alignment_jobs = []
    for info in alignment_input_jobs:
        name = info['name']
        uncropped_fielded_fpath = info['uncropped_fielded_fpath']
        aligned_imgonly_fpath = info['aligned_imgonly_fpath']
        aligned_imganns_fpath = info['aligned_imganns_fpath']
        region_globstr = info['region_globstr']
        site_globstr = info['site_globstr']
        parent_node = info['node']

        debug_valid_regions = config.debug
        align_visualize = config.debug
        include_channels = config.include_channels
        exclude_channels = config.exclude_channels

        # MAIN WORKHORSE CROP IMAGES
        # Crop big images to the geojson regions
        import kwutil
        if config.sensor_to_time_window:
            sensor_to_time_window = kwutil.util_yaml.Yaml.dumps(kwutil.util_yaml.Yaml.loads(config.sensor_to_time_window))
            sensor_to_time_window = "\n" + ub.indent(sensor_to_time_window, ' ' * (4 * 6))
        else:
            sensor_to_time_window = config.sensor_to_time_window
        align_node = new_pipeline.submit(
            name=f'align-geotiffs-{name}',
            executable=ub.codeblock(
                fr'''
                {job_environ_str}python -m geowatch.cli.coco_align \
                    --regions "{region_globstr}" \
                    --context_factor=1 \
                    --geo_preprop=auto \
                    --keep={config.align_keep} \
                    --force_nodata={config.force_nodata} \
                    --include_channels="{include_channels}" \
                    --exclude_channels="{exclude_channels}" \
                    --visualize={align_visualize} \
                    --debug_valid_regions={debug_valid_regions} \
                    --rpc_align_method {config.rpc_align_method} \
                    --sensor_to_time_window "{sensor_to_time_window}" \
                    --verbose={config.verbose} \
                    --aux_workers={config.align_aux_workers} \
                    --target_gsd="{config.target_gsd}" \
                    --force_min_gsd="{config.force_min_gsd}" \
                    --workers={config.align_workers} \
                    --tries="{config.align_tries}" \
                    --cooldown="10" \
                    --backoff="3" \
                    --asset_timeout="{config.asset_timeout}" \
                    --image_timeout="{config.image_timeout}" \
                    --hack_lazy={config.hack_lazy}
                '''),
            in_paths=_justkeys({
                'src': uncropped_fielded_fpath,
            }),
            out_paths={
                'dst': aligned_imgonly_fpath,
                'dst_bundle_dpath': aligned_kwcoco_bundle,
            },
            group_dname=aligned_bundle_name,
        )
        parent_node.outputs['dst'].connect(align_node.inputs['src'])

        if config['visualize']:
            aligned_img_viz_dpath = aligned_kwcoco_bundle / '_viz512_img'
            viz_max_dim = 512
            viz_img_node = new_pipeline.submit(
                name=f'viz-imgs-{name}',
                executable=ub.codeblock(
                    fr'''
                    python -m geowatch visualize \
                        --draw_anns=False \
                        --draw_imgs=True \
                        --channels="red|green|blue" \
                        --max_dim={viz_max_dim} \
                        --stack=only \
                        --animate=True \
                        --workers=auto
                    '''),
                in_paths=_justkeys({
                    'src': aligned_imgonly_fpath,
                }),
                out_paths={
                    'viz_dpath': aligned_img_viz_dpath,
                },
                group_dname=aligned_bundle_name,
            )
            align_node.outputs['dst'].connect(viz_img_node.inputs['src'])

        if site_globstr and config.reproject_annotations:
            # Visualization here is too slow, add on another option if we
            # really need to
            viz_part = ''
            project_anns_node = new_pipeline.submit(
                name=f'project-annots-{name}',
                executable=ub.codeblock(
                    r'''
                    python -m geowatch reproject_annotations \
                        --propogate_strategy="{config.propogate_strategy}" \
                        --site_models="{site_globstr}" \
                        --io_workers="avail/2" \
                        --region_models="{region_globstr}" {viz_part}
                    ''').format(**locals()),
                in_paths=_justkeys({
                    'src': aligned_imgonly_fpath,
                }),
                out_paths={
                    'dst': aligned_imganns_fpath,
                },
                group_dname=aligned_bundle_name,
            )
            align_node.outputs['dst'].connect(project_anns_node.inputs['src'])

            if config.visualize:
                aligned_img_viz_dpath = aligned_kwcoco_bundle / '_viz512_ann'
                viz_ann_node = new_pipeline.submit(
                    name=f'viz-annots-{name}',
                    executable=ub.codeblock(
                        fr'''
                        python -m geowatch visualize \
                            --draw_anns=True \
                            --draw_imgs=False \
                            --channels="red|green|blue" \
                            --max_dim={viz_max_dim} \
                            --animate=True \
                            --workers=auto \
                            --stack=only \
                            --only_boxes={config["visualize_only_boxes"]}
                        '''),
                    in_paths=_justkeys({
                        'src': aligned_imganns_fpath,
                    }),
                    out_paths={
                        'viz_dpath': aligned_img_viz_dpath,
                    },
                    group_dname=aligned_bundle_name,
                )
                project_anns_node.outputs['dst'].connect(viz_ann_node.inputs['src'])
        else:
            aligned_imganns_fpath = aligned_imgonly_fpath
            info['aligned_imganns_fpath'] = aligned_imgonly_fpath
            project_anns_node = align_node

        align_info = info.copy()
        align_info['node'] = project_anns_node
        alignment_jobs.append(align_info)

    # Need a final union step
    aligned_fpaths = [d['aligned_imganns_fpath'] for d in alignment_jobs]
    union_depends_nodes = [d['node'] for d in alignment_jobs]

    aligned_final_fpath = (aligned_kwcoco_bundle / 'data.kwcoco.zip')
    aligned_multi_src_part = ' '.join(['"{}"'.format(p) for p in aligned_fpaths])

    # COMBINE Uncropped datasets
    if config.final_union:
        union_node = new_pipeline.submit(
            name='kwcoco-union',
            executable=ub.codeblock(
                fr'''
                {job_environ_str}python -m kwcoco union
                '''),
            in_paths=_justkeys({
                'src': aligned_fpaths,
            }),
            out_paths={
                'dst': aligned_final_fpath,
            },
            group_dname=aligned_bundle_name,
        )
        for node in union_depends_nodes:
            node.outputs['dst'].connect(union_node.inputs['src'])

        aligned_final_nodes = [union_node]

    # Determine what stages will be cached.
    cache = config.cache
    if isinstance(cache, str):
        cache = [p.strip() for p in cache.split(',')]

    config.tmux_workers = min(len(stac_jobs), config.tmux_workers)
    queue = config.create_queue(environ=environ)

    new_pipeline.build_nx_graphs()

    # new_pipeline.inspect_configurables()
    rich.print('pipe_config = {}'.format(ub.urepr(pipe_config, nl=1)))

    new_pipeline.configure(
        config=pipe_config,
        root_dpath=out_dpath,
        cache=config.cache
    )

    # new_pipeline.inspect_configurables()
    # new_pipeline.print_graphs()

    new_pipeline.submit_jobs(
        queue, skip_existing=config.skip_existing,
        enable_links=False,
        write_configs=1,
        write_invocations=1,
    )

    # queue.print_graph()
    # queue.rprint()

    # Do Basic Splits
    if config.splits:
        # Note: we probably could just do unions more cleverly rather than
        # splitting.
        raise NotImplementedError('broken')
        from geowatch.cli import prepare_splits
        prepare_splits._submit_split_jobs(
            aligned_final_fpath, queue, depends=[aligned_final_nodes])

        # TODO: use this instead
        # prepare_splits._submit_constructive_split_jobs(
        #     base_fpath, dst_dpath, suffix, queue, config
        # )

    # TODO: Can start the DVC add of the region subdirectories here
    ub.codeblock(
        '''
        7z a splits.zip data*.kwcoco.zip

        ls */WV
        ls */L8
        ls */S2
        ls */*.json
        dvc add *.zip
        dvc add */WV */L8 */S2 */*.json *.zip

        DVC_DPATH=$(geowatch_dvc)
        echo "DVC_DPATH='$DVC_DPATH'"

        cd $DVC_DPATH/
        git pull  # ensure you are up to date with master on DVC
        cd $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10
        dvc pull */L8.dvc */S2.dvc
        dvc pull
        */*.json
        ''')

    print_kwargs = {
        'with_status': 0,
        'style': "colors",
        'with_locks': 0,
        'exclude_tags': ['boilerplate'],
    }
    config.run_queue(queue, system=True, print_kwargs=print_kwargs)
    # if config.rprint:
    #     queue.print_graph()
    #     queue.rprint()
    # if config.run:
    #     # This logic will exist in cmd-queue itself
    #     other_session_handler = config.other_session_handler
    #     queue.run(block=True, system=True, with_textual=config.with_textual,
    #               other_session_handler=other_session_handler)

    # TODO: team features
    """
    DATASET_CODE=Aligned-Drop2-TA1-2022-03-07
    DVC_DPATH=$(geowatch_dvc)
    DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    python -m geowatch.cli.prepare_teamfeats \
        --base_fpath=$KWCOCO_BUNDLE_DPATH/data.kwcoco.zip \
        --gres=0, \
        --with_depth=0 \
        --with_landcover=0 \
        --with_invariants=0 \
        --with_materials=1 \
        --depth_workers=auto \
        --do_splits=1  --cache=1 --run=0
    """

# dvc_dpath=$home/data/dvc-repos/smart_watch_dvc
# #dvc_dpath=$(geowatch_dvc)
# #s3_dpath=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working
# query_basename=$(basename "$s3_fpath")
# aligned_bundle_name=aligned-$dataset_suffix
# uncropped_bundle_name=uncropped-$dataset_suffix

# #region_models=$dvc_dpath'/annotations/region_models/*.geojson'
# region_models=$dvc_dpath/annotations/region_models/kr_r002.geojson
# # helper variables
# uncropped_dpath=$dvc_dpath/$uncropped_bundle_name
# uncropped_query_dpath=$uncropped_dpath/_query/items
# uncropped_ingress_dpath=$uncropped_dpath/ingress
# uncropped_kwcoco_fpath=$uncropped_dpath/data.kwcoco.zip
# aligned_kwcoco_bundle=$dvc_dpath/$aligned_bundle_name
# aligned_kwcoco_fpath=$aligned_kwcoco_bundle/data.kwcoco.zip
# uncropped_query_fpath=$uncropped_query_dpath/$query_basename
# uncropped_catalog_fpath=$uncropped_ingress_dpath/catalog.json

# export AWS_DEFAULT_PROFILE=iarpa
#     pass


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/prepare_ta2_dataset.py
    """
    main(cmdline=True)

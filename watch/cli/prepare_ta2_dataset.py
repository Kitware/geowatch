r"""
An end-to-end script for calling all the scripts needed to

* Pulls the STAC catalog that points to processed large image tiles
* Creates a virtual Uncropped kwcoco dataset that points to the large image tiles
* Crops the dataset to create an aligned TA2 dataset

See Also:
    ~/code/watch/scripts/prepare_drop3.sh
    ~/code/watch/scripts/prepare_drop4.sh
    ~/code/watch/scripts/prepare_drop5.sh

Example:

    # Create a demo region file, and create vairables that point at relevant
    # paths, which are by default written in your ~/.cache folder
    xdoctest -m watch.demo.demo_region demo_khq_region_fpath
    REGION_FPATH="$HOME/.cache/watch/demo/annotations/KHQ_R001.geojson"
    SITE_GLOBSTR="$HOME/.cache/watch/demo/annotations/KHQ_R001_sites/*.geojson"

    # The "name" of the new dataset
    DATASET_SUFFIX=Demo-TA2-KHQ

    # Set this to where you want to build the dataset
    DEMO_DPATH=$PWD/prep_ta2_demo

    mkdir -p "$DEMO_DPATH"

    # This is a string code indicating what STAC endpoint we will pull from
    SENSORS="sentinel-s2-l2a-cogs"

    # Depending on the STAC endpoint, some parameters may need to change:
    # collated - True for IARPA endpoints, Usually False for public data
    # requester_pays - True for public landsat
    # api_key - A secret for non public data

    export SMART_STAC_API_KEY=""
    export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
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
        --convert_workers=8 \
        --align_workers=26 \
        --cache=0 \
        --ignore_duplicates=1 \
        --target_gsd=30 \
        --visualize=True \
        --max_products_per_region=10 \
        --serial=True --run=0

    smartwatch visualize $HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/data.kwcoco_c9ea8bb9.json

"""


import scriptconfig as scfg
import ubelt as ub
import warnings


class PrepareTA2Config(scfg.Config):
    default = {
        'dataset_suffix': scfg.Value(None, help=''),

        'stac_query_mode': scfg.Value(None, help='if set to auto we try to make the .input files. Mutex with s3_fpath'),
        'cloud_cover': scfg.Value(10, help='maximum cloud cover percentage (ignored if s3_fpath given)'),
        'sensors': scfg.Value("L2", help='(ignored if s3_fpath given)'),
        'max_products_per_region': scfg.Value(None, help='does uniform affinity sampling over time to filter down to this many results per region'),
        'api_key': scfg.Value('env:SMART_STAC_API_KEY', help='The API key or where to get it (ignored if s3_fpath given)'),

        'separate_region_queues': scfg.Value(True, help='if True, create jobs for each region separately. This option to disable this may be removed in the future.'),
        'separate_align_jobs': scfg.Value(True, help='if True, perform alignment for each region in its own job. The option to disable this may be removed in the future.'),

        's3_fpath': scfg.Value(None, nargs='+', help='A list of .input files which were the results of an existing stac query. Mutex with stac_query_* args. Mutex with sensors.'),

        'out_dpath': scfg.Value('auto', help='This is the path that all resulting files will be written to. Defaults the the phase2 DATA_DVC_DPATH', alias=['dvc_dpath']),

        'run': scfg.Value('0', isflag=1, help='if True execute the pipeline'),
        'collated': scfg.Value([True], nargs='+', help='set to false if the input data is not collated'),

        'backend': scfg.Value('serial', help='can be serial, tmux, or slurm. Using tmux is recommended.'),
        'serial': scfg.Value(False, isflag=True, help='if True, override other settings to disable parallelism. DEPRECATE. Set backend=serial instead'),
        'with_textual': scfg.Value('auto', help='setting for cmd-queue monitoring'),
        'other_session_handler': scfg.Value('ask', help='for tmux backend only. How to handle conflicting sessions. Can be ask, kill, or ignore, or auto'),
        'queue_name': scfg.Value('prep-ta2-dataset', help='name for the command queue'),
        'rprint': scfg.Value(False, isflag=True, help='enable rich printing of the commands'),

        'max_queue_size': scfg.Value(10, help='the number of regions allowed to be processed in parallel with tmux backend'),
        'max_regions': None,

        'aws_profile': scfg.Value('iarpa', help='AWS profile to use for remote data access'),

        'query_workers': scfg.Value('0', help='workers for STAC search'),
        'convert_workers': scfg.Value('min(avail,8)', help='workers for stac-to-kwcoco script'),
        'fields_workers': scfg.Value('min(avail,max(all/2,8))', help='workers for add-watch-fields script'),
        'align_workers': scfg.Value(0, help='primary workers for align script'),
        'align_aux_workers': scfg.Value(0, help='threads per align process (typically set this to 0)'),

        'ignore_duplicates': scfg.Value(1, help='workers for align script'),

        'visualize': scfg.Value(0, isflag=1, help='if True runs visualize'),
        'visualize_only_boxes': scfg.Value(True, isflag=1, help='if False will draw full polygons'),

        'verbose': scfg.Value(0, help='help control verbosity (just align for now)'),

        # '--requester_pays'
        'requester_pays': scfg.Value(0, help='if True, turn on requester_pays in ingress. Needed for official L1/L2 catalogs.'),

        'debug': scfg.Value(False, isflag=1, help='if enabled, turns on debug visualizations'),
        'select_images': scfg.Value(False, help='if enabled only uses select images'),

        'cache': scfg.Value(1, isflag=1, help='If 1 or 0 globally enable/disable caching. If a comma separated list of strings, only cache those stages'),

        'include_channels': scfg.Value(None, help='specific channels to use in align crop'),
        'exclude_channels': scfg.Value(None, help='specific channels to NOT use in align crop'),

        'splits': scfg.Value(False, isflag=1, help='if True do splits'),

        'region_globstr': scfg.Value('annotations/region_models', help='region model globstr (relative to the dvc path, unless absolute or prefixed by "./")'),
        'site_globstr': scfg.Value('annotations/site_models', help='site model globstr (relative to the dvc path, unless absolute or prefixed by "./")'),

        'propogate_strategy': scfg.Value('SMART', help='changes propogation behavior'),

        'target_gsd': 10,
        'remove_broken': scfg.Value(True, isflag=1, help='if True, will remove any image that fails population (e.g. caused by a 404)'),
        'force_nodata': scfg.Value(None, help='if specified, forces nodata to this value'),

        'align_keep': scfg.Value('img', choices=['img', 'img-roi', 'none', None], help='if the coco align script caches or recomputes images / rois'),

        'skip_existing': scfg.Value(False, help='Unlike cache=1, which checks for file existence at runtime, this will explicitly not submit any job with a product that already exist'),

        'rpc_align_method': scfg.Value('orthorectify', help=ub.paragraph(
            '''
            Can be one of:
                (1) orthorectify - which uses gdalwarp with -rpc if available
                    otherwise falls back to affine transform,
                (2) affine_warp - which ignores RPCs and uses the affine
                    transform in the geotiff metadata.
            '''
        )),

        'hack_lazy': scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Hack lazy is a proof of concept with the intent on speeding up the
            download / cropping of data by flattening the gdal processing into
            a single queue of parallel processes executed via a command queue.

            By running once with this flag on, it will execute the command
            queue, and then running again, it should see all of the data as
            existing and construct the aligned kwcoco dataset as normal.
            ''')),
    }


__config__ = PrepareTA2Config


def main(cmdline=False, **kwargs):
    """
    Example:
        from watch.cli.prepare_ta2_dataset import *  # NOQA
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
    # import shlex
    config = PrepareTA2Config(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    if config['serial']:
        config['backend'] = 'serial'
        config['convert_workers'] = 0
        config['fields_workers'] = 0
        config['align_workers'] = 0
        config['align_aux_workers'] = 0

    out_dpath = config['out_dpath']
    if out_dpath == 'auto':
        import watch
        out_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    out_dpath = ub.Path(out_dpath)

    aws_profile = config['aws_profile']

    aligned_bundle_name = f'Aligned-{config["dataset_suffix"]}'
    uncropped_bundle_name = f'Uncropped-{config["dataset_suffix"]}'

    uncropped_dpath = out_dpath / uncropped_bundle_name
    uncropped_query_dpath = uncropped_dpath / '_query/items'

    uncropped_ingress_dpath = uncropped_dpath / 'ingress'

    aligned_kwcoco_bundle = out_dpath / aligned_bundle_name

    uncropped_dpath = uncropped_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_ingress_dpath = uncropped_ingress_dpath.shrinkuser(home='$HOME')
    aligned_kwcoco_bundle = aligned_kwcoco_bundle.shrinkuser(home='$HOME')

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

    def _coerce_globstr(p):
        if not p:
            return None
        globstr = ub.Path(p)
        if str(globstr).startswith('./'):
            final_globstr = globstr
        else:
            final_globstr = out_dpath / globstr
        final_globstr = final_globstr.shrinkuser(home='$HOME')
        return final_globstr

    # region_models = list(region_dpath.glob('*.geojson'))
    final_region_globstr = _coerce_globstr(config['region_globstr'])
    final_site_globstr = _coerce_globstr(config['site_globstr'])

    import cmd_queue
    from watch.utils import util_gis

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
    from watch.mlops import pipeline
    pipeline = pipeline.Pipeline()

    stac_jobs = []
    # base_mkdir_job = pipeline.submit(f'mkdir -p "{uncropped_query_dpath}"', name='mkdir-base')
    if config['stac_query_mode'] == 'auto':
        # Each region gets their own job in the queue
        if config['separate_region_queues']:

            # Note: this requires the annotation files to exist on disk.  or we
            # have to write a mechanism that lets the explicit relative path be
            # specified.
            region_file_fpaths = util_gis.coerce_geojson_paths(final_region_globstr.expand())

            if final_site_globstr:
                region_site_fpaths = util_gis.coerce_geojson_paths(final_site_globstr.expand())
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
                    print(f'regions_without_sites={regions_without_sites}')
                    print(f'sites_without_regions={sites_without_regions}')

            else:
                raise NotImplementedError(
                    'TODO: implement more robust alternative that reads '
                    'file data to make assignment if needed')

            if config['max_regions'] is not None:
                region_file_fpaths = region_file_fpaths[:config['max_regions']]

            print('region_file_fpaths = {}'.format(ub.repr2(sorted(region_file_fpaths), nl=1)))
            for region_id, region_fpath in region_id_to_fpath.items():
                if region_id in region_id_blocklist:
                    continue

                region_inputs_fpath = (uncropped_query_dpath / (region_id + '.input')).shrinkuser(home='$HOME')
                final_region_fpath = region_fpath.shrinkuser(home='$HOME')

                stac_search_job = pipeline.submit(
                    command=ub.codeblock(
                        rf'''
                        python -m watch.cli.stac_search \
                            --region_file "{final_region_fpath}" \
                            --search_json "auto" \
                            --cloud_cover "{config['cloud_cover']}" \
                            --sensors "{config['sensors']}" \
                            --api_key "{config['api_key']}" \
                            --query_workers = "{config['query_workers']}" \
                            --max_products_per_region "{config['max_products_per_region']}" \
                            --append_mode=False \
                            --mode area \
                            --verbose 2 \
                            --outfile "{region_inputs_fpath}"
                        '''),
                    name=f'stac-search-{region_id}',
                    depends=[],
                    in_paths={
                        'final_region_fpath': final_region_fpath,
                    },
                    out_paths={
                        'region_inputs_fpath': region_inputs_fpath,
                    },
                    stage='stac_search',
                )
                # cache_prefix = f'[[ -f {region_inputs_fpath} ]] || ' if stages.nodes['stac_search']['cache'] else ''

                stac_jobs.append({
                    'name': region_id,
                    'job': stac_search_job,
                    'inputs_fpath': region_inputs_fpath,
                    'region_globstr': final_region_fpath,
                    'collated': default_collated,
                })
        else:
            warnings.warn(
                'It is usually faster to split the queue amongst regions')
            # All region queries are executed simultaniously and put into a
            # single inputs file. The advantage here is we dont need to know
            # how many regions there are beforehand.
            combined_inputs_fpath = (uncropped_query_dpath / (f'combo_query_{config["dataset_suffix"]}.input')).shrinkuser(home='$HOME')

            combo_stac_search_job = pipeline.submit(
                command=ub.codeblock(
                    rf'''
                    python -m watch.cli.stac_search \
                    --region_globstr "{final_region_globstr}" \
                    --search_json "auto" \
                    --cloud_cover "{config['cloud_cover']}" \
                    --sensors "{config['sensors']}" \
                    --api_key "{config['api_key']}" \
                    --max_products_per_region "{config['max_products_per_region']}" \
                    --append_mode=False \
                    --mode area \
                    --verbose 2 \
                    --outfile "{combined_inputs_fpath}"
                    '''),
                name='stac-search', depends=[],
                in_paths={
                    'final_region_globstr': final_region_globstr,
                },
                out_paths={
                    'combined_inputs_fpath': combined_inputs_fpath,
                },
                stage='stac_search',
            )

            stac_jobs.append({
                'name': 'combined',
                'job': combo_stac_search_job,
                'region_globstr': final_region_globstr,
                'inputs_fpath': combined_inputs_fpath,
                'collated': default_collated,
            })
    else:
        s3_fpath_list = config['s3_fpath']
        collated_list = config['collated']
        if len(collated_list) != len(s3_fpath_list):
            print('Indicate if each s3 path is collated or not')
        stac_jobs = []
        for s3_fpath, collated in zip(s3_fpath_list, collated_list):
            stac_jobs.append({
                'job': None,
                'name': ub.Path(s3_fpath).stem,
                'inputs_fpath': s3_fpath,
                'region_globstr': final_region_globstr,
                'collated': collated,
            })

    uncropped_fielded_jobs = []
    for stac_job in stac_jobs:
        s3_fpath = ub.Path(stac_job['inputs_fpath'])
        s3_name = stac_job['name']
        parent_job = stac_job['job']
        collated = stac_job['collated']
        uncropped_query_fpath = uncropped_query_dpath / s3_fpath.name
        uncropped_query_fpath = uncropped_query_fpath.shrinkuser(home='$HOME')

        uncropped_catalog_fpath = uncropped_ingress_dpath / f'catalog_{s3_name}.json'
        uncropped_ingress_dpath = uncropped_ingress_dpath.shrinkuser(home='$HOME')

        if not str(s3_fpath).startswith('s3'):
            # Don't really need to copy anything in this case.
            uncropped_query_fpath = s3_fpath
            grab_job = parent_job
            # grab_job = pipeline.submit(ub.codeblock(
            #     f'''
            #     # GRAB Input STAC List
            #     {cache_prefix}cp "{s3_fpath}" "{uncropped_query_dpath}"
            #     '''), depends=parent_job, name=f'psudo-s3-pull-inputs-{s3_name}')
        else:
            grab_job = pipeline.submit(command=ub.codeblock(
                f'''
                # GRAB Input STAC List
                aws s3 --profile {aws_profile} cp "{s3_fpath}" "{uncropped_query_dpath}"
                '''), depends=parent_job, name=f's3-pull-inputs-{s3_name}',
                in_paths={
                    's3_fpath': s3_fpath,
                },
                out_paths={
                    'uncropped_query_fpath': uncropped_query_fpath,
                },
                stage='grab',
            )

        ingress_options = [
            '--virtual',
        ]
        if config['requester_pays']:
            ingress_options.append('--requester_pays')
        ingress_options_str = ' '.join(ingress_options)

        ingress_job = pipeline.submit(command=ub.codeblock(
            rf'''
            python -m watch.cli.baseline_framework_ingress \
                --aws_profile {aws_profile} \
                --jobs avail \
                {ingress_options_str} \
                --outdir "{uncropped_ingress_dpath}" \
                --catalog_fpath "{uncropped_catalog_fpath}" \
                "{uncropped_query_fpath}"
            '''), depends=[grab_job], name=f'baseline_ingress-{s3_name}',
            in_paths={
                'uncropped_query_fpath': uncropped_query_fpath,
            },
            out_paths={
                'uncropped_catalog_fpath': uncropped_catalog_fpath,
            },
            stage='catalog',
        )

        uncropped_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}.kwcoco.json'
        uncropped_kwcoco_fpath = uncropped_kwcoco_fpath.shrinkuser(home='$HOME')

        convert_options = []
        if collated:
            convert_options.append('--from-collated')
        if config['ignore_duplicates']:
            convert_options.append('--ignore_duplicates')
        convert_options_str = ' '.join(convert_options)

        convert_job = pipeline.submit(
            command=ub.codeblock(
                rf'''
                {job_environ_str}python -m watch.cli.ta1_stac_to_kwcoco \
                    "{uncropped_catalog_fpath}" \
                    --outpath="{uncropped_kwcoco_fpath}" \
                    {convert_options_str} \
                    --jobs "{config['convert_workers']}"
                '''),
            depends=[ingress_job],
            name=f'ta1_stac_to_kwcoco-{s3_name}',
            in_paths={
                'uncropped_catalog_fpath': uncropped_catalog_fpath,
            },
            out_paths={
                'uncropped_kwcoco_fpath': uncropped_kwcoco_fpath,
            },
            stage='uncropped_kwcoco',
        )

        uncropped_fielded_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}_fielded.kwcoco.json'
        uncropped_fielded_kwcoco_fpath = uncropped_fielded_kwcoco_fpath.shrinkuser(home='$HOME')

        add_fields_job = pipeline.submit(command=ub.codeblock(
            rf'''
            {job_environ_str}python -m watch.cli.coco_add_watch_fields \
                --src "{uncropped_kwcoco_fpath}" \
                --dst "{uncropped_fielded_kwcoco_fpath}" \
                --enable_video_stats=False \
                --overwrite=warp \
                --target_gsd={config['target_gsd']} \
                --remove_broken={config['remove_broken']} \
                --workers="{config['fields_workers']}"
            '''),
            depends=convert_job, name=f'coco_add_watch_fields-{s3_name}',
            in_paths={
                'uncropped_kwcoco_fpath': uncropped_kwcoco_fpath,
            },
            out_paths={
                'uncropped_fielded_kwcoco_fpath': uncropped_fielded_kwcoco_fpath,
            },
            stage='uncropped_feilds',
        )

        uncropped_fielded_jobs.append({
            'name': stac_job['name'],
            'job': add_fields_job,
            'uncropped_fielded_fpath': uncropped_fielded_kwcoco_fpath,
            'region_globstr': stac_job['region_globstr'],
        })

    if config['separate_align_jobs']:
        alignment_input_jobs = []
        # TODO: make use of region_id_to_site_fpaths
        # to build a better site globstr (might need to write a tempoaray file
        # to point to)
        final_site_globstr = _coerce_globstr(config['site_globstr'])
        for info in uncropped_fielded_jobs:
            toalign_info = info.copy()
            name = toalign_info['name'] = info['name']
            toalign_info['aligned_imgonly_fpath'] = aligned_kwcoco_bundle / f'imgonly-{name}.kwcoco.json'
            toalign_info['aligned_imganns_fpath'] = aligned_kwcoco_bundle / f'imganns-{name}.kwcoco.json'
            # TODO: take only the corresponding set of site models here.
            toalign_info['site_globstr'] = final_site_globstr
            toalign_info['region_globstr'] = info['region_globstr']
            alignment_input_jobs.append(toalign_info)
    else:
        # Do a noop for this case? Or make it fast in kwcoco itself?
        uncropped_coco_paths = [d['uncropped_fielded_fpath'] for d in uncropped_fielded_jobs]
        union_depends_jobs = [d['job'] for d in uncropped_fielded_jobs]
        union_suffix = ub.hash_data([p.name for p in uncropped_coco_paths])[0:8]
        uncropped_final_kwcoco_fpath = uncropped_dpath / f'data_{union_suffix}.kwcoco.json'
        uncropped_final_kwcoco_fpath = uncropped_final_kwcoco_fpath.shrinkuser(home='$HOME')
        uncropped_multi_src_part = ' '.join(['"{}"'.format(p) for p in uncropped_coco_paths])
        union_job = pipeline.submit(command=ub.codeblock(
            rf'''
            # COMBINE Uncropped datasets
            {job_environ_str}python -m kwcoco union \
                --src {uncropped_multi_src_part} \
                --dst "{uncropped_final_kwcoco_fpath}"
            '''), depends=union_depends_jobs, name='kwcoco-union',
            in_paths={
                'uncropped_coco_paths': uncropped_coco_paths,
            },
            out_paths={
                'uncropped_final_kwcoco_fpath': uncropped_final_kwcoco_fpath,
            },
            stage='union_uncropped_feilds',
        )
        uncropped_final_jobs = [union_job]

        final_site_globstr = _coerce_globstr(config['site_globstr'])
        alignment_input_jobs = [{
            'name': f'align-{union_suffix}',
            'uncropped_fielded_fpath': uncropped_final_kwcoco_fpath,
            'aligned_imgonly_fpath': aligned_kwcoco_bundle / 'imgonly.kwcoco.json',
            'aligned_imganns_fpath': aligned_kwcoco_bundle / 'imganns.kwcoco.json',
            'region_globstr': final_region_globstr,
            'site_globstr': final_site_globstr,
            'job': uncropped_final_jobs,
        }]

    alignment_jobs = []
    for info in alignment_input_jobs:
        name = info['name']
        uncropped_fielded_fpath = info['uncropped_fielded_fpath']
        aligned_imgonly_fpath = info['aligned_imgonly_fpath']
        aligned_imganns_fpath = info['aligned_imganns_fpath']
        region_globstr = info['region_globstr']
        site_globstr = info['site_globstr']
        parent_job = info['job']

        # cache_crops = 1
        # if cache_crops:
        #     align_keep = 'img'
        #     # align_keep = 'roi-img'
        # else:
        #     align_keep = 'none'

        debug_valid_regions = config['debug']
        align_visualize = config['debug']
        include_channels = config['include_channels']
        exclude_channels = config['exclude_channels']

        align_job = pipeline.submit(command=ub.codeblock(
            rf'''
            # MAIN WORKHORSE CROP IMAGES
            # Crop big images to the geojson regions
            {job_environ_str}python -m watch.cli.coco_align_geotiffs \
                --src "{uncropped_fielded_fpath}" \
                --dst "{aligned_imgonly_fpath}" \
                --regions "{region_globstr}" \
                --context_factor=1 \
                --geo_preprop=auto \
                --keep={config['align_keep']} \
                --force_nodata={config['force_nodata']} \
                --include_channels="{include_channels}" \
                --exclude_channels="{exclude_channels}" \
                --visualize={align_visualize} \
                --debug_valid_regions={debug_valid_regions} \
                --rpc_align_method {config['rpc_align_method']} \
                --verbose={config['verbose']} \
                --aux_workers={config['align_aux_workers']} \
                --target_gsd={config['target_gsd']} \
                --workers={config['align_workers']} \
                --hack_lazy={config['hack_lazy']}
            '''),
            depends=parent_job,
            name=f'align-geotiffs-{name}',
            in_paths={
                'uncropped_fielded_fpath': uncropped_fielded_fpath,
            },
            out_paths={
                'aligned_imgonly_fpath': aligned_imgonly_fpath,
            },
            stage='align_kwcoco',
        )

        # TODO:
        # Project annotation from latest annotations subdir
        # Prepare splits
        # Add baseline datasets to DVC

        aligned_viz_dpath = aligned_kwcoco_bundle / '_viz512'
        viz_max_dim = 512

        if config['visualize']:
            pipeline.submit(command=ub.codeblock(
                rf'''
                python -m watch visualize \
                    --src "{aligned_imgonly_fpath}" \
                    --viz_dpath "{aligned_viz_dpath}" \
                    --draw_anns=False \
                    --draw_imgs=True \
                    --channels="red|green|blue" \
                    --max_dim={viz_max_dim} \
                    --stack=only \
                    --animate=True --workers=auto
                '''), depends=[align_job], name=f'viz-imgs-{name}',
                in_paths={
                    'aligned_imgonly_fpath': aligned_imgonly_fpath,
                },
                out_paths={
                    'aligned_viz_dpath': aligned_viz_dpath,
                },
                stage='viz_imgs',
            )

        if site_globstr:
            # site_model_dpath = (dvc_dpath / 'annotations/site_models').shrinkuser(home='$HOME')
            # region_model_dpath = (dvc_dpath / 'annotations/region_models').shrinkuser(home='$HOME')

            # Visualization here is too slow, add on another option if we really
            # need to
            # viz_part = '--viz_dpath=auto' if config['visualize'] else ''
            viz_part = ''
            project_anns_job = pipeline.submit(command=ub.codeblock(
                rf'''
                python -m watch project_annotations \
                    --src "{aligned_imgonly_fpath}" \
                    --dst "{aligned_imganns_fpath}" \
                    --propogate_strategy="{config['propogate_strategy']}" \
                    --site_models="{site_globstr}" \
                    --region_models="{region_globstr}" {viz_part}
                '''), depends=[align_job], name=f'project-annots-{name}',
                in_paths={
                    'aligned_imgonly_fpath': aligned_imgonly_fpath,
                },
                out_paths={
                    'aligned_imganns_fpath': aligned_imganns_fpath,
                },
                stage='project_annots',
            )

            if config['visualize']:
                pipeline.submit(command=ub.codeblock(
                    rf'''
                    python -m watch visualize \
                        --src "{aligned_imganns_fpath}" \
                        --viz_dpath "{aligned_viz_dpath}" \
                        --draw_anns=True \
                        --draw_imgs=False \
                        --channels="red|green|blue" \
                        --max_dim={viz_max_dim} \
                        --animate=True --workers=auto \
                        --stack=only \
                        --only_boxes={config["visualize_only_boxes"]}
                    '''), depends=[project_anns_job], name=f'viz-annots-{name}',
                    in_paths={
                        'aligned_imganns_fpath': aligned_imganns_fpath,
                    },
                    out_paths={
                        'aligned_viz_dpath': aligned_viz_dpath,
                    },
                    stage='viz_anns',
                )
        else:
            aligned_imganns_fpath = aligned_imgonly_fpath
            info['aligned_imganns_fpath'] = aligned_imgonly_fpath
            project_anns_job = align_job

        align_info = info.copy()
        align_info['job'] = project_anns_job
        alignment_jobs.append(align_info)

    if config['separate_align_jobs']:
        # Need a final union step
        aligned_fpaths = [d['aligned_imganns_fpath'] for d in alignment_jobs]
        union_depends_jobs = [d['job'] for d in alignment_jobs]
        # union_suffix = ub.hash_data([p.name for p in aligned_fpaths])[0:8]
        aligned_final_fpath = (aligned_kwcoco_bundle / 'data.kwcoco.json').shrinkuser(home='$HOME')
        aligned_multi_src_part = ' '.join(['"{}"'.format(p) for p in aligned_fpaths])
        # cache_prefix = f'[[ -f {aligned_final_fpath} ]] || ' if stages.nodes['final_union']['cache'] else ''
        union_job = pipeline.submit(command=ub.codeblock(
            rf'''
            # COMBINE Uncropped datasets
            {job_environ_str}python -m kwcoco union \
                --src {aligned_multi_src_part} \
                --dst "{aligned_final_fpath}"
            '''), depends=union_depends_jobs, name='kwcoco-union',
            in_paths={
                'aligned_fpaths': aligned_fpaths,
            },
            out_paths={
                'aligned_final_fpath': aligned_final_fpath,
            },
            stage='final_union',
        )
        aligned_final_jobs = [union_job]
    else:
        assert len(alignment_jobs) == 1
        aligned_final_fpath = alignment_jobs[0]['aligned_imganns_fpath']
        aligned_final_fpath = alignment_jobs[0]['aligned_imganns_fpath']
        aligned_final_jobs = [alignment_jobs[0]['job']]

    # TODO:
    # queue.synchronize -
    # force all submissions to finish before starting new ones.

    # Determine what stages will be cached.
    cache = config['cache']
    if isinstance(cache, str):
        cache = [p.strip() for p in cache.split(',')]

    self = pipeline
    pipeline._update_stage_otf_cache(cache)

    queue = cmd_queue.Queue.create(
        backend=config['backend'], name=config['queue_name'], size=1,
        gres=None, environ=environ)

    # hack to set number of parallel sessions based on job size
    queue.size = min(len(stac_jobs), config['max_queue_size'])

    # self._populate_explicit_dependency_queue(queue)
    self._populate_implicit_dependency_queue(
        queue, skip_existing=config['skip_existing'])

    # queue.print_graph()
    # queue.rprint()

    # Do Basic Splits
    if config['splits']:
        # Note: we probably could just do unions more cleverly rather than
        # splitting.
        from watch.cli import prepare_splits
        prepare_splits._submit_split_jobs(
            aligned_final_fpath, queue, depends=[aligned_final_jobs])

    # TODO: Can start the DVC add of the region subdirectories here
    ub.codeblock(
        '''
        7z a splits.zip data*.kwcoco.json

        ls */WV
        ls */L8
        ls */S2
        ls */*.json
        dvc add *.zip
        dvc add */WV */L8 */S2 */*.json *.zip

        DVC_DPATH=$(smartwatch_dvc)
        echo "DVC_DPATH='$DVC_DPATH'"

        cd $DVC_DPATH/
        git pull  # ensure you are up to date with master on DVC
        cd $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10
        dvc pull */L8.dvc */S2.dvc
        dvc pull
        */*.json
        ''')

    # pipeline.submit(ub.codeblock(
    #     '''
    #     DVC_DPATH=$(smartwatch_dvc)
    #     python -m watch.cli.prepare_splits \
    #         --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
    #         --run=1 --serial=True
    #     '''))

    if config['rprint']:
        queue.print_graph()
        queue.rprint()

    if config['run']:

        # This logic will exist in cmd-queue itself
        other_session_handler = config['other_session_handler']

        def handle_other_sessions(other_session_handler):
            if other_session_handler == 'auto':
                from cmd_queue.tmux_queue import has_stdin
                if has_stdin():
                    other_session_handler = 'ask'
                else:
                    other_session_handler = 'kill'
            if other_session_handler == 'ask':
                queue.kill_other_queues(ask_first=True)
            elif other_session_handler == 'kill':
                queue.kill_other_queues(ask_first=False)
            elif other_session_handler == 'ignore':
                ...
            else:
                raise KeyError

        if config['backend'] == 'tmux':
            handle_other_sessions(other_session_handler)
        queue.run(block=True, system=True, with_textual=config['with_textual'],
                  check_other_sessions=False)

    # TODO: team features
    """
    DATASET_CODE=Aligned-Drop2-TA1-2022-03-07
    DVC_DPATH=$(smartwatch_dvc)
    DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    python -m watch.cli.prepare_teamfeats \
        --base_fpath=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
        --gres=0, \
        --with_depth=0 \
        --with_landcover=0 \
        --with_invariants=0 \
        --with_materials=1 \
        --depth_workers=auto \
        --do_splits=1  --cache=1 --run=0
    """

# dvc_dpath=$home/data/dvc-repos/smart_watch_dvc
# #dvc_dpath=$(smartwatch_dvc)
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
# uncropped_kwcoco_fpath=$uncropped_dpath/data.kwcoco.json
# aligned_kwcoco_bundle=$dvc_dpath/$aligned_bundle_name
# aligned_kwcoco_fpath=$aligned_kwcoco_bundle/data.kwcoco.json
# uncropped_query_fpath=$uncropped_query_dpath/$query_basename
# uncropped_catalog_fpath=$uncropped_ingress_dpath/catalog.json

# export AWS_DEFAULT_PROFILE=iarpa
#     pass


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/prepare_ta2_dataset.py
    """
    main(cmdline=True)

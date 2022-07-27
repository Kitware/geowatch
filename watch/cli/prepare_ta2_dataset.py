r"""
An end-to-end script for calling all the scripts needed to

* Pulls the STAC catalog that points to processed large image tiles
* Creates a virtual Uncropped kwcoco dataset that points to the large image tiles
* Crops the dataset to create an aligned TA2 dataset

See Also:
    ~/code/watch/scripts/prepare_drop3.sh
    ~/code/watch/scripts/prepare_drop4.sh


TODO:
    - [ ] Rename to schedule_ta2_dataset


Examples:

DATASET_SUFFIX=TA1_FULL_SEQ_KR_S001_CLOUD_LT_10
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working/KR_S001.unique.fixed_ls_ids.cloudcover_lt_10.output



DVC_DPATH=$(smartwatch_dvc)
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input
DATASET_SUFFIX=Drop2-TA1-2022-02-24


S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input.l1.mini



DVC_DPATH=$(smartwatch_dvc)
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/big-stac-file-on-aws
DATASET_SUFFIX=my-dataset-name
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath=$S3_FPATH \
    --dvc_dpath=$DVC_DPATH \
    --collated=True \
    --requester_pays=True \
    --ignore_duplicates=True \
    --fields_workers=0 \
    --align_workers=0 \
    --convert_workers=0 \
    --debug=False \
    --run=0 --cache=False

        --select_images '.id % 1200 == 0'  \


DVC_DPATH=$(smartwatch_dvc)
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220120/iMERIT_COMBINED.unique.input
DATASET_SUFFIX=Drop2-TA1-2022-03-07
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath=$S3_FPATH \
    --dvc_dpath=$DVC_DPATH \
    --collated=False \
    --align_workers=4 \
    --run=0


jq .images[0] $HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/data.kwcoco_c9ea8bb9.json

kwcoco visualize $HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/data.kwcoco_c9ea8bb9.json

"""


import scriptconfig as scfg
import ubelt as ub
import warnings


class PrepareTA2Config(scfg.Config):
    default = {
        'dataset_suffix': scfg.Value(None, help=''),

        'stac_query_mode': scfg.Value(None, help='if set to auto we try to make the .input files (ignored if s3_fpath given)'),
        'cloud_cover': scfg.Value(10, help='maximum cloud cover percentage (ignored if s3_fpath given)'),
        'sensors': scfg.Value("L2", help='(ignored if s3_fpath given)'),
        'max_products_per_region': scfg.Value(None, help='does uniform affinity sampling over time to filter down to this many results per region'),
        'api_key': scfg.Value('env:SMART_STAC_API_KEY', help='The API key or where to get it (ignored if s3_fpath given)'),

        'separate_region_queues': scfg.Value(True, help='if True, create jobs for each region separately'),
        'separate_align_jobs': scfg.Value(True, help='if True, perform alignment for each region in its own job'),

        's3_fpath': scfg.Value(None, nargs='+', help='A list of .input files which were the results of an existing stac query. Mutex with stac_query_* args'),
        'dvc_dpath': scfg.Value('auto', help=''),
        'run': scfg.Value('0', help='if True execute the pipeline'),
        'collated': scfg.Value([True], nargs='+', help='set to false if the input data is not collated'),

        'backend': scfg.Value('serial', help='can be serial, tmux, or slurm. Using tmux is recommended.'),
        'max_queue_size': scfg.Value(10, help='the number of regions allowed to be processed in parallel with tmux backend'),
        'max_regions': None,

        'aws_profile': scfg.Value('iarpa', help='AWS profile to use for remote data access'),

        'convert_workers': scfg.Value('min(avail,8)', help='workers for stac-to-kwcoco script'),
        'fields_workers': scfg.Value('min(avail,max(all/2,8))', help='workers for add-watch-fields script'),
        'align_workers': scfg.Value(0, help='workers for align script'),
        'align_aux_workers': scfg.Value(0, help='threads per align process (typically set this to 0)'),

        'ignore_duplicates': scfg.Value(1, help='workers for align script'),

        'visualize': scfg.Value(0, help='if True runs visualize'),

        'verbose': scfg.Value(0, help='help control verbosity (just align for now)'),

        # '--requester_pays'
        'requester_pays': scfg.Value(0, help='if True, turn on requester_pays in ingress. Needed for official L1/L2 catalogs.'),

        'debug': scfg.Value(False, help='if enabled, turns on debug visualizations'),
        'select_images': scfg.Value(False, help='if enabled only uses select images'),

        'cache': scfg.Value(1, help='if enabled check cache'),

        'include_channels': scfg.Value(None, help='specific channels to use in align crop'),
        'exclude_channels': scfg.Value(None, help='specific channels to NOT use in align crop'),

        'splits': scfg.Value(False, help='if True do splits'),

        'region_globstr': scfg.Value('annotations/region_models', help='region model globstr (relative to the dvc path, unless absolute or prefixed by "./")'),
        'site_globstr': scfg.Value('annotations/site_models', help='site model globstr (relative to the dvc path, unless absolute or prefixed by "./")'),

        'target_gsd': 10,
        'remove_broken': scfg.Value(True, help='if True, will remove any image that fails population (e.g. caused by a 404)')
    }


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

    dvc_dpath = config['dvc_dpath']
    if dvc_dpath == 'auto':
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
    dvc_dpath = ub.Path(dvc_dpath)

    aws_profile = config['aws_profile']

    aligned_bundle_name = f'Aligned-{config["dataset_suffix"]}'
    uncropped_bundle_name = f'Uncropped-{config["dataset_suffix"]}'

    uncropped_dpath = dvc_dpath / uncropped_bundle_name
    uncropped_query_dpath = uncropped_dpath / '_query/items'

    uncropped_ingress_dpath = uncropped_dpath / 'ingress'

    aligned_kwcoco_bundle = dvc_dpath / aligned_bundle_name

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
        globstr = ub.Path(p)
        if str(globstr).startswith('./'):
            final_globstr = globstr
        else:
            final_globstr = dvc_dpath / globstr
        final_globstr = final_globstr.shrinkuser(home='$HOME')
        return final_globstr

    # region_models = list(region_dpath.glob('*.geojson'))
    final_region_globstr = _coerce_globstr(config['region_globstr'])
    final_site_globstr = _coerce_globstr(config['site_globstr'])

    import cmd_queue
    from watch.utils import util_path
    queue = cmd_queue.Queue.create(
        backend=config['backend'], name='prep-ta2-dataset', size=1, gres=None)

    default_collated = config['collated'][0]

    stac_jobs = []
    # base_mkdir_job = queue.submit(f'mkdir -p "{uncropped_query_dpath}"', name='mkdir-base')
    if config['stac_query_mode'] == 'auto':
        # Each region gets their own job in the queue
        if config['separate_region_queues']:

            # Note: this requires the annotation files to exist on disk.  or we
            # have to write a mechanism that lets the explicit relative path be
            # specified.
            region_file_fpaths = util_path.coerce_patterned_paths(final_region_globstr.expand())
            region_site_fpaths = util_path.coerce_patterned_paths(final_site_globstr.expand())

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

                cache_prefix = f'[[ -f {region_inputs_fpath} ]] || ' if config['cache'] else ''
                stac_search_job = queue.submit(ub.codeblock(
                    rf'''
                    {cache_prefix}python -m watch.cli.stac_search \
                        --region_file "{final_region_fpath}" \
                        --search_json "auto" \
                        --cloud_cover "{config['cloud_cover']}" \
                        --sensors "{config['sensors']}" \
                        --api_key "{config['api_key']}" \
                        --max_products_per_region "{config['max_products_per_region']}" \
                        --mode area \
                        --verbose 2 \
                        --outfile "{region_inputs_fpath}"
                    '''), name=f'stac-search-{region_id}', depends=[])

                stac_jobs.append({
                    'name': region_id,
                    'job': stac_search_job,
                    'inputs_fpath': region_inputs_fpath,
                    'region_globstr': final_region_fpath,
                    'collated': default_collated,
                })

            # Kind of pointless option, we could separate all stac jobs and
            # then combine them, not sure if we want to be able to do that
            # though. We could do it for a subset if we wanted.
            if False and len(stac_jobs) > 1:
                # Combine all into a single path.
                input_fpath_list = [d['inputs_fpath'] for d in stac_jobs]
                jobs = [d['job'] for d in stac_jobs]
                combo_hash = ub.hash_data(stac_jobs)[0:8]
                combo_name = f'combo_{combo_hash}'
                combined_inputs_fpath = (uncropped_query_dpath / (f'{combo_name}.input')).shrinkuser(home='$HOME')
                quoted_fpath_list = ['"{}"'.format(p) for p in input_fpath_list]
                combine_job = queue.submit(ub.codeblock(
                    f'''
                    # GRAB Input STAC List
                    cat {' '.join(quoted_fpath_list)} > "{combined_inputs_fpath}"
                    '''), depends=jobs, name=combo_name)
                stac_jobs = [{
                    'name': combo_name,
                    'job': combine_job,
                    'inputs_fpath': combined_inputs_fpath,
                    'region_globstr': final_region_globstr,
                    'collated': default_collated,
                }]
        else:
            warnings.warn(
                'It is usually faster to split the queue amongst regions')
            # All region queries are executed simultaniously and put into a
            # single inputs file. The advantage here is we dont need to know
            # how many regions there are beforehand.
            combined_inputs_fpath = (uncropped_query_dpath / (f'combo_query_{config["dataset_suffix"]}.input')).shrinkuser(home='$HOME')
            combo_stac_search_job = queue.submit(ub.codeblock(
                rf'''
                python -m watch.cli.stac_search \
                    --region_globstr "{final_region_globstr}" \
                    --search_json "auto" \
                    --cloud_cover "{config['cloud_cover']}" \
                    --sensors "{config['sensors']}" \
                    --api_key "{config['api_key']}" \
                    --max_products_per_region "{config['max_products_per_region']}" \
                    --mode area \
                    --verbose 2 \
                    --outfile "{combined_inputs_fpath}"
                '''), name='stac-search', depends=[])

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

    # hack to dynamically resize tmux jobs
    queue.size = min(len(stac_jobs), config['max_queue_size'])

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

        cache_prefix = f'[[ -f {uncropped_query_fpath} ]] || ' if config['cache'] else ''
        if not str(s3_fpath).startswith('s3'):
            # Don't really need to copy anything in this case.
            uncropped_query_fpath = s3_fpath
            grab_job = parent_job
            # grab_job = queue.submit(ub.codeblock(
            #     f'''
            #     # GRAB Input STAC List
            #     {cache_prefix}cp "{s3_fpath}" "{uncropped_query_dpath}"
            #     '''), depends=parent_job, name=f'psudo-s3-pull-inputs-{s3_name}')
        else:
            grab_job = queue.submit(ub.codeblock(
                f'''
                # GRAB Input STAC List
                {cache_prefix}aws s3 --profile {aws_profile} cp "{s3_fpath}" "{uncropped_query_dpath}"
                '''), depends=parent_job, name=f's3-pull-inputs-{s3_name}')

        ingress_options = [
            '--virtual',
        ]
        if config['requester_pays']:
            ingress_options.append('--requester_pays')
        ingress_options_str = ' '.join(ingress_options)

        cache_prefix = f'[[ -f {uncropped_catalog_fpath} ]] || ' if config['cache'] else ''
        ingress_job = queue.submit(ub.codeblock(
            rf'''
            {cache_prefix}python -m watch.cli.baseline_framework_ingress \
                --aws_profile {aws_profile} \
                --jobs avail \
                {ingress_options_str} \
                --outdir "{uncropped_ingress_dpath}" \
                --catalog_fpath "{uncropped_catalog_fpath}" \
                "{uncropped_query_fpath}"
            '''), depends=[grab_job], name=f'baseline_ingress-{s3_name}')

        uncropped_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}.kwcoco.json'
        uncropped_kwcoco_fpath = uncropped_kwcoco_fpath.shrinkuser(home='$HOME')

        convert_options = []
        if collated:
            convert_options.append('--from-collated')
        if config['ignore_duplicates']:
            convert_options.append('--ignore_duplicates')
        convert_options_str = ' '.join(convert_options)

        cache_prefix = f'[[ -f {uncropped_kwcoco_fpath} ]] || ' if config['cache'] else ''
        convert_job = queue.submit(ub.codeblock(
            rf'''
            {cache_prefix}{job_environ_str}python -m watch.cli.ta1_stac_to_kwcoco \
                "{uncropped_catalog_fpath}" \
                --outpath="{uncropped_kwcoco_fpath}" \
                {convert_options_str} \
                --jobs "{config['convert_workers']}"
            '''), depends=[ingress_job], name=f'ta1_stac_to_kwcoco-{s3_name}')

        uncropped_fielded_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}_fielded.kwcoco.json'
        uncropped_fielded_kwcoco_fpath = uncropped_fielded_kwcoco_fpath.shrinkuser(home='$HOME')

        cache_prefix = f'[[ -f {uncropped_fielded_kwcoco_fpath} ]] || ' if config['cache'] else ''
        add_fields_job = queue.submit(ub.codeblock(
            rf'''
            # PREPARE Uncropped datasets
            {cache_prefix}{job_environ_str}python -m watch.cli.coco_add_watch_fields \
                --src "{uncropped_kwcoco_fpath}" \
                --dst "{uncropped_fielded_kwcoco_fpath}" \
                --enable_video_stats=False \
                --overwrite=warp \
                --target_gsd={config['target_gsd']} \
                --remove_broken={config['remove_broken']} \
                --workers="{config['fields_workers']}"
            '''), depends=convert_job, name=f'coco_add_watch_fields-{s3_name}')

        uncropped_fielded_jobs.append({
            'name': stac_job['name'],
            'job': add_fields_job,
            'uncropped_fielded_fpath': uncropped_fielded_kwcoco_fpath,
            'region_globstr': stac_job['region_globstr'],
        })

    if config['separate_align_jobs']:
        alignment_input_jobs = []
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
        cache_prefix = f'[[ -f {uncropped_final_kwcoco_fpath} ]] || ' if config['cache'] else ''
        union_job = queue.submit(ub.codeblock(
            rf'''
            # COMBINE Uncropped datasets
            {cache_prefix}{job_environ_str}python -m kwcoco union \
                --src {uncropped_multi_src_part} \
                --dst "{uncropped_final_kwcoco_fpath}"
            '''), depends=union_depends_jobs, name='kwcoco-union')
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

        cache_crops = 1
        if cache_crops:
            align_keep = 'img'
            # align_keep = 'roi-img'
        else:
            align_keep = 'none'

        debug_valid_regions = config['debug']
        align_visualize = config['debug']
        include_channels = config['include_channels']
        exclude_channels = config['exclude_channels']

        align_job = queue.submit(ub.codeblock(
            rf'''
            # MAIN WORKHORSE CROP IMAGES
            # Crop big images to the geojson regions
            {job_environ_str}python -m watch.cli.coco_align_geotiffs \
                --src "{uncropped_fielded_fpath}" \
                --dst "{aligned_imgonly_fpath}" \
                --regions "{region_globstr}" \
                --context_factor=1 \
                --geo_preprop=auto \
                --keep={align_keep} \
                --include_channels="{include_channels}" \
                --exclude_channels="{exclude_channels}" \
                --visualize={align_visualize} \
                --debug_valid_regions={debug_valid_regions} \
                --rpc_align_method affine_warp \
                --verbose={config['verbose']} \
                --aux_workers={config['align_aux_workers']} \
                --workers={config['align_workers']}
            '''), depends=parent_job, name=f'align-geotiffs-{name}')

        # TODO:
        # Project annotation from latest annotations subdir
        # Prepare splits
        # Add baseline datasets to DVC

        aligned_viz_dpath = aligned_kwcoco_bundle / '_viz512'
        viz_max_dim = 512

        if config['visualize']:
            queue.submit(ub.codeblock(
                rf'''
                # Update to whatever the state of the annotations submodule is
                python -m watch visualize \
                    --src "{aligned_imgonly_fpath}" \
                    --viz_dpath "{aligned_viz_dpath}" \
                    --draw_anns=False \
                    --draw_imgs=True \
                    --channels="red|green|blue" \
                    --max_dim={viz_max_dim} \
                    --animate=True --workers=auto
                '''), depends=[align_job], name=f'viz-imgs-{name}')

        if 1:
            # site_model_dpath = (dvc_dpath / 'annotations/site_models').shrinkuser(home='$HOME')
            # region_model_dpath = (dvc_dpath / 'annotations/region_models').shrinkuser(home='$HOME')

            # Visualization here is too slow, add on another option if we really
            # need to
            # viz_part = '--viz_dpath=auto' if config['visualize'] else ''
            viz_part = ''
            cache_prefix = f'[[ -f {aligned_imganns_fpath} ]] || ' if config['cache'] else ''
            project_anns_job = queue.submit(ub.codeblock(
                rf'''
                # Update to whatever the state of the annotations submodule is
                {cache_prefix}python -m watch project_annotations \
                    --src "{aligned_imgonly_fpath}" \
                    --dst "{aligned_imganns_fpath}" \
                    --site_models="{site_globstr}" \
                    --region_models="{region_globstr}" {viz_part}
                '''), depends=[align_job], name=f'project-annots-{name}')

        if config['visualize']:
            queue.submit(ub.codeblock(
                rf'''
                # Update to whatever the state of the annotations submodule is
                python -m watch visualize \
                    --src "{aligned_imganns_fpath}" \
                    --viz_dpath "{aligned_viz_dpath}" \
                    --draw_anns=True \
                    --draw_imgs=False \
                    --channels="red|green|blue" \
                    --max_dim={viz_max_dim} \
                    --animate=True --workers=auto \
                    --only_boxes=True
                '''), depends=[project_anns_job], name=f'viz-annots-{name}')

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
        cache_prefix = f'[[ -f {aligned_final_fpath} ]] || ' if config['cache'] else ''
        union_job = queue.submit(ub.codeblock(
            rf'''
            # COMBINE Uncropped datasets
            {cache_prefix}{job_environ_str}python -m kwcoco union \
                --src {aligned_multi_src_part} \
                --dst "{aligned_final_fpath}"
            '''), depends=union_depends_jobs, name='kwcoco-union')
        aligned_final_jobs = [union_job]
    else:
        assert len(alignment_jobs) == 1
        aligned_final_fpath = alignment_jobs[0]['aligned_imganns_fpath']
        aligned_final_fpath = alignment_jobs[0]['aligned_imganns_fpath']
        aligned_final_jobs = [alignment_jobs[0]['job']]

    # TODO:
    # queue.synchronize -
    # force all submissions to finish before starting new ones.

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

    # queue.submit(ub.codeblock(
    #     '''
    #     DVC_DPATH=$(smartwatch_dvc)
    #     python -m watch.cli.prepare_splits \
    #         --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
    #         --run=1 --serial=True
    #     '''))

    queue.rprint()
    queue.print_graph()
    if config['run']:
        queue.run(block=True, system=True)

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

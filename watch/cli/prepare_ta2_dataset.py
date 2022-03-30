"""
An end-to-end script for calling all the scripts needed to

* Pulls the STAC catalog that points to processed large image tiles
* Creates a virtual Uncropped kwcoco dataset that points to the large image tiles
* Crops the dataset to create an aligned TA2 dataset

See Also:
    ~/code/watch/scripts/prepare_drop3.sh


TODO:
    - [ ] Rename to schedule_ta2_dataset


Examples:

DATASET_SUFFIX=TA1_FULL_SEQ_KR_S001_CLOUD_LT_10
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working/KR_S001.unique.fixed_ls_ids.cloudcover_lt_10.output



DVC_DPATH=$(python -m watch.cli.find_dvc)
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input
DATASET_SUFFIX=Drop2-TA1-2022-02-24


DVC_DPATH=$(python -m watch.cli.find_dvc)
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input.l1.mini
DATASET_SUFFIX=foobar

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
    --serial=True --run=0 --cache=False

        --select_images '.id % 1200 == 0'  \


DVC_DPATH=$(python -m watch.cli.find_dvc)
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220120/iMERIT_COMBINED.unique.input
DATASET_SUFFIX=Drop2-TA1-2022-03-07
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath=$S3_FPATH \
    --dvc_dpath=$DVC_DPATH \
    --collated=False \
    --align_workers=4 \
    --serial=True --run=0


jq .images[0] $HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/data.kwcoco_c9ea8bb9.json

kwcoco visualize $HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/data.kwcoco_c9ea8bb9.json

"""


import scriptconfig as scfg
import ubelt as ub


class PrepareTA2Config(scfg.Config):
    default = {
        'dataset_suffix': scfg.Value(None, help=''),
        's3_fpath': scfg.Value(None, nargs='+', help=''),
        'dvc_dpath': scfg.Value('auto', help=''),
        'run': scfg.Value('0', help=''),
        'collated': scfg.Value([True], nargs='+', help='set to false if the input data is not collated'),

        'backend': scfg.Value('serial', help='can be serial, tmux, or slurm'),

        'aws_profile': scfg.Value('iarpa', help='AWS profile to use for remote data access'),

        'convert_workers': scfg.Value('min(avail,8)', help='workers for stac-to-kwcoco script'),
        'fields_workers': scfg.Value('min(avail,max(all/2,8))', help='workers for add-watch-fields script'),
        'align_workers': scfg.Value(0, help='workers for align script'),
        'align_aux_workers': scfg.Value(0, help='threads per align process (typically set this to 0)'),

        'ignore_duplicates': scfg.Value(0, help='workers for align script'),

        'visualize': scfg.Value(0, help='if True runs visualize'),

        'verbose': scfg.Value(0, help='help control verbosity (just align for now)'),

        # '--requester_pays'
        'requester_pays': scfg.Value(0, help='if True, turn on requester_pays in ingress. Needed for official L1/L2 catalogs.'),

        'debug': scfg.Value(False, help='if enabled, turns on debug visualizations'),
        'select_images': scfg.Value(False, help='if enabled only uses select images'),

        'cache': scfg.Value(1, help='if enabled check cache'),

        'channels': scfg.Value(None, help='specific channels to use in align crop'),
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

    region_dpath = dvc_dpath / 'annotations/region_models'
    # region_models = list(region_dpath.glob('*.geojson'))

    uncropped_dpath = dvc_dpath / uncropped_bundle_name
    uncropped_query_dpath = uncropped_dpath / '_query/items'

    uncropped_ingress_dpath = uncropped_dpath / 'ingress'

    aligned_kwcoco_bundle = dvc_dpath / aligned_bundle_name
    aligned_imgonly_kwcoco_fpath = aligned_kwcoco_bundle / 'imgonly.kwcoco.json'
    aligned_imganns_kwcoco_fpath = aligned_kwcoco_bundle / 'data.kwcoco.json'

    region_dpath = region_dpath.shrinkuser(home='$HOME')
    uncropped_dpath = uncropped_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_ingress_dpath = uncropped_ingress_dpath.shrinkuser(home='$HOME')
    aligned_kwcoco_bundle = aligned_kwcoco_bundle.shrinkuser(home='$HOME')
    aligned_imgonly_kwcoco_fpath = aligned_imgonly_kwcoco_fpath.shrinkuser(home='$HOME')
    aligned_imganns_kwcoco_fpath = aligned_imganns_kwcoco_fpath.shrinkuser(home='$HOME')

    from watch.utils import cmd_queue
    queue = cmd_queue.Queue.create(
        backend=config['backend'], name='teamfeat', size=1, gres=None)

    s3_fpath_list = config['s3_fpath']
    collated_list = config['collated']
    if len(collated_list) != len(s3_fpath_list):
        print('Indicate if each s3 path is collated or not')

    job_environs = [
        # 'PROJ_DEBUG=3',
        f'AWS_DEFAULT_PROFILE={aws_profile}',
    ]
    if config['requester_pays']:
        job_environs.append("AWS_REQUEST_PAYER='requester'")
    job_environ_str = ' '.join(job_environs)
    if job_environ_str:
        job_environ_str += ' '

    uncropped_coco_paths = []
    union_depends_jobs = []
    for s3_fpath, collated in zip(s3_fpath_list, collated_list):
        s3_name = ub.Path(s3_fpath).name
        uncropped_query_fpath = uncropped_query_dpath / ub.Path(s3_fpath).name
        uncropped_query_fpath = uncropped_query_fpath.shrinkuser(home='$HOME')

        uncropped_catalog_fpath = uncropped_ingress_dpath / f'catalog_{s3_name}.json'
        uncropped_ingress_dpath = uncropped_ingress_dpath.shrinkuser(home='$HOME')

        cache_prefix = '[[ -f {uncropped_query_fpath} ]] || ' if config['cache'] else ''
        grab_job = queue.submit(ub.codeblock(
            f'''
            # GRAB Input STAC List
            mkdir -p {uncropped_query_dpath}
            {cache_prefix}aws s3 --profile {aws_profile} cp "{s3_fpath}" "{uncropped_query_dpath}"
            '''))

        ingress_options = [
            '--virtual',
        ]
        if config['requester_pays']:
            ingress_options.append('--requester_pays')
        ingress_options_str = ' '.join(ingress_options)

        cache_prefix = '[[ -f {uncropped_catalog_fpath} ]] || ' if config['cache'] else ''
        ingress_job = queue.submit(ub.codeblock(
            rf'''
            {cache_prefix}python -m watch.cli.baseline_framework_ingress \
                --aws_profile {aws_profile} \
                --jobs avail \
                {ingress_options_str} \
                --outdir "{uncropped_ingress_dpath}" \
                --catalog_fpath "{uncropped_catalog_fpath}" \
                "{uncropped_query_fpath}"
            '''), depends=[grab_job])

        uncropped_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}.kwcoco.json'
        uncropped_kwcoco_fpath = uncropped_kwcoco_fpath.shrinkuser(home='$HOME')

        convert_options = []
        if collated:
            convert_options.append('--from-collated')
        if config['ignore_duplicates']:
            convert_options.append('--ignore_duplicates')
        convert_options_str = ' '.join(convert_options)

        cache_prefix = '[[ -f {uncropped_kwcoco_fpath} ]] || ' if config['cache'] else ''
        convert_job = queue.submit(ub.codeblock(
            rf'''
            {cache_prefix}{job_environ_str}python -m watch.cli.ta1_stac_to_kwcoco \
                "{uncropped_catalog_fpath}" \
                --outpath="{uncropped_kwcoco_fpath}" \
                {convert_options_str} \
                --jobs "{config['convert_workers']}"
            '''), depends=[ingress_job])

        uncropped_fielded_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}_fielded.kwcoco.json'
        uncropped_fielded_kwcoco_fpath = uncropped_fielded_kwcoco_fpath.shrinkuser(home='$HOME')

        cache_prefix = '[[ -f {uncropped_fielded_kwcoco_fpath} ]] || ' if config['cache'] else ''
        add_fields_job = queue.submit(ub.codeblock(
            rf'''
            # PREPARE Uncropped datasets (usually for debugging)
            {cache_prefix}{job_environ_str}python -m watch.cli.coco_add_watch_fields \
                --src "{uncropped_kwcoco_fpath}" \
                --dst "{uncropped_fielded_kwcoco_fpath}" \
                --enable_video_stats=False \
                --overwrite=warp \
                --target_gsd=10 \
                --workers="{config['fields_workers']}"
            '''), depends=convert_job)

        uncropped_coco_paths.append(uncropped_fielded_kwcoco_fpath)
        union_depends_jobs.append(add_fields_job)

    if len(uncropped_coco_paths) == 1:
        uncropped_final_kwcoco_fpath = uncropped_coco_paths[0]
        uncropped_final_jobs = union_depends_jobs
    else:
        union_suffix = ub.hash_data([p.name for p in uncropped_coco_paths])[0:8]
        uncropped_final_kwcoco_fpath = uncropped_dpath / f'data_{union_suffix}.kwcoco.json'
        uncropped_final_kwcoco_fpath = uncropped_final_kwcoco_fpath.shrinkuser(home='$HOME')
        uncropped_multi_src_part = ' '.join(['"{}"'.format(p) for p in uncropped_coco_paths])
        cache_prefix = '[[ -f {uncropped_final_kwcoco_fpath} ]] || ' if config['cache'] else ''
        union_job = queue.submit(ub.codeblock(
            rf'''
            # COMBINE Uncropped datasets
            {cache_prefix}{job_environ_str}python -m kwcoco union \
                --src {uncropped_multi_src_part} \
                --dst "{uncropped_final_kwcoco_fpath}"
            '''), depends=union_depends_jobs)
        uncropped_final_jobs = [union_job]

    # uncropped_prep_kwcoco_fpath = uncropped_dpath / 'data_prepped.kwcoco.json'
    # uncropped_prep_kwcoco_fpath = uncropped_prep_kwcoco_fpath.shrinkuser(home='$HOME')
    # select_images_query = config['select_images']
    # if select_images_query:
    #     suffix = '_' + ub.hash_data(select_images_query)[0:8]
    #     # Debugging
    #     small_uncropped_kwcoco_fpath = uncropped_kwcoco_fpath.augment(suffix=suffix)
    #     subset_job = queue.submit(ub.codeblock(
    #         rf'''
    #         # SUBSET Uncropped datasets (usually for debugging)
    #         python -m kwcoco subset \
    #             --src="{uncropped_final_kwcoco_fpath}" \
    #             --dst="{small_uncropped_kwcoco_fpath}" \
    #             --select_images='{select_images_query}'
    #         '''), jobs=uncropped_final_jobs)
    #     # --populate-watch-fields \
    #     add_fields_depends = [subset_job]
    #     uncropped_final_kwcoco_fpath = small_uncropped_kwcoco_fpath
    #     uncropped_prep_kwcoco_fpath = uncropped_prep_kwcoco_fpath.augment(suffix=suffix)
    #     aligned_imgonly_kwcoco_fpath = aligned_imgonly_kwcoco_fpath.augment(suffix=suffix)
    # else:
    # add_fields_depends = uncropped_final_jobs

    # region_model_str = ' '.join([shlex.quote(str(p)) for p in region_models])

    cache_crops = 1
    if cache_crops:
        align_keep = 'img'
        align_keep = 'roi-img'
    else:
        align_keep = 'none'

    debug_valid_regions = config['debug']
    align_visualize = config['debug']
    channels = config['channels']
    align_job = queue.submit(ub.codeblock(
        rf'''
        # MAIN WORKHORSE CROP IMAGES
        # Crop big images to the geojson regions
        {job_environ_str}python -m watch.cli.coco_align_geotiffs \
            --src "{uncropped_final_kwcoco_fpath}" \
            --dst "{aligned_imgonly_kwcoco_fpath}" \
            --regions "{region_dpath / '*.geojson'}" \
            --context_factor=1 \
            --geo_preprop=auto \
            --keep={align_keep} \
            --channels="{channels}" \
            --visualize={align_visualize} \
            --debug_valid_regions={debug_valid_regions} \
            --rpc_align_method affine_warp \
            --verbose={config['verbose']} \
            --aux_workers={config['align_aux_workers']} \
            --workers={config['align_workers']}
        '''), depends=uncropped_final_jobs)

    # TODO:
    # Project annotation from latest annotations subdir
    # Prepare splits
    # Add baseline datasets to DVC

    if config['visualize']:
        queue.submit(ub.codeblock(
            rf'''
            # Update to whatever the state of the annotations submodule is
            python -m watch visualize \
                --src "{aligned_imgonly_kwcoco_fpath}" \
                --draw_anns=False \
                --draw_imgs=True \
                --channels="red|green|blue" \
                --animate=True --workers=auto
            '''), depends=[align_job])

    if 1:
        site_model_dpath = (dvc_dpath / 'annotations/site_models').shrinkuser(home='$HOME')
        region_model_dpath = (dvc_dpath / 'annotations/region_models').shrinkuser(home='$HOME')

        viz_part = '--viz_dpath=auto' if config['visualize'] else ''
        project_anns_job = queue.submit(ub.codeblock(
            rf'''
            # Update to whatever the state of the annotations submodule is
            python -m watch project_annotations \
                --src "{aligned_imgonly_kwcoco_fpath}" \
                --dst "{aligned_imganns_kwcoco_fpath}" \
                --site_models="{site_model_dpath}/*.geojson" \
                --region_models="{region_model_dpath}/*.geojson" {viz_part}
            '''), depends=[align_job])

    # TODO:
    # queue.synchronize -
    # force all submissions to finish before starting new ones.

    # Do Basic Splits
    if 1:
        from watch.cli import prepare_splits
        prepare_splits._submit_split_jobs(
            aligned_imganns_kwcoco_fpath, queue, depends=[project_anns_job])

    # TODO: Can start the DVC add of the region subdirectories here
    ub.codeblock(
        '''
        cd /home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop3-TA1-2022-03-10
        ls */WV
        ls */L8
        ls */S2
        ls */*.json

        dvc add */WV */L8 */S2 */*.json
        dvc add data_*nowv*.kwcoco.json

        DVC_DPATH=$(python -m watch.cli.find_dvc)
        echo "DVC_DPATH='$DVC_DPATH'"

        cd $DVC_DPATH/
        git pull  # ensure you are up to date with master on DVC
        cd $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10
        dvc pull */L8.dvc */S2.dvc
        dvc pull
        */*.json

        ''')

    if config['visualize']:
        queue.submit(ub.codeblock(
            rf'''
            # Update to whatever the state of the annotations submodule is
            python -m watch visualize \
                --src "{aligned_imganns_kwcoco_fpath}" \
                --draw_anns=True \
                --draw_imgs=False \
                --channels="red|green|blue" \
                --animate=True --workers=auto
            '''), depends=[project_anns_job])

    # queue.submit(ub.codeblock(
    #     '''
    #     DVC_DPATH=$(python -m watch.cli.find_dvc)
    #     python -m watch.cli.prepare_splits \
    #         --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
    #         --run=1 --serial=True
    #     '''))

    queue.rprint()

    if config['run']:
        agg_state = None
        # if config['serial']:
        #     queue.serial_run()
        # else:
        queue.run()
        # if config['follow']:
        agg_state = queue.monitor()
        # if not config['keep_sessions']:
        if agg_state is not None and not agg_state['errored']:
            queue.kill()

    # TODO: team features
    """
    DATASET_CODE=Aligned-Drop2-TA1-2022-03-07
    DVC_DPATH=$(python -m watch.cli.find_dvc)
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
# #dvc_dpath=$(python -m watch.cli.find_dvc)
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

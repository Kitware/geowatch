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
    --debug=False \
    --align_workers=0 \
    --serial=True --run=0

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
        'serial': scfg.Value(True, help='if True use serial mode'),
        'aws_profile': scfg.Value('iarpa', help='AWS profile to use for remote data access'),
        'align_workers': scfg.Value(0, help='workers for align script'),

        'ignore_duplicates': scfg.Value(0, help='workers for align script'),

        'visualize': scfg.Value(0, help='if True runs visualize'),

        # '--requester_pays'
        'requester_pays': scfg.Value(0, help='if True, turn on requester_pays in ingress'),

        'debug': scfg.Value(False, help='if enabled, turns on debug visualizations'),
        'select_images': scfg.Value(False, help='if enabled only uses select images')
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
    from watch.utils import tmux_queue
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
    aligned_kwcoco_fpath = aligned_kwcoco_bundle / 'data.kwcoco.json'

    region_dpath = region_dpath.shrinkuser(home='$HOME')
    uncropped_dpath = uncropped_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_ingress_dpath = uncropped_ingress_dpath.shrinkuser(home='$HOME')
    aligned_kwcoco_bundle = aligned_kwcoco_bundle.shrinkuser(home='$HOME')
    aligned_kwcoco_fpath = aligned_kwcoco_fpath.shrinkuser(home='$HOME')

    # queue = tmux_queue.SerialQueue()
    queue = tmux_queue.TMUXMultiQueue(name='teamfeat', size=1, gres=None)

    s3_fpath_list = config['s3_fpath']
    collated_list = config['collated']
    if len(collated_list) != len(s3_fpath_list):
        print('Indicate if each s3 path is collated or not')

    uncropped_coco_paths = []
    for s3_fpath, collated in zip(s3_fpath_list, collated_list):
        s3_name = ub.Path(s3_fpath).name
        uncropped_query_fpath = uncropped_query_dpath / ub.Path(s3_fpath).name
        uncropped_query_fpath = uncropped_query_fpath.shrinkuser(home='$HOME')

        uncropped_catalog_fpath = uncropped_ingress_dpath / f'catalog_{s3_name}.json'
        uncropped_ingress_dpath = uncropped_ingress_dpath.shrinkuser(home='$HOME')

        queue.submit(ub.codeblock(
            f'''
            # GRAB Input STAC List
            mkdir -p {uncropped_query_dpath}
            [[ -f {uncropped_query_fpath} ]] || aws s3 --profile {aws_profile} cp "{s3_fpath}" "{uncropped_query_dpath}"
            '''))

        ingress_options = [
            '--virtual',
        ]
        if config['requester_pays']:
            ingress_options.append('--requester_pays')
        ingress_options_str = ' '.join(ingress_options)

        queue.submit(ub.codeblock(
            rf'''
            [[ -f {uncropped_catalog_fpath} ]] || python -m watch.cli.baseline_framework_ingress \
                --aws_profile {aws_profile} \
                --jobs avail \
                {ingress_options_str} \
                --outdir "{uncropped_ingress_dpath}" \
                --catalog_fpath "{uncropped_catalog_fpath}" \
                "{uncropped_query_fpath}"
            '''))

        uncropped_kwcoco_fpath = uncropped_dpath / f'data_{s3_name}.kwcoco.json'
        uncropped_kwcoco_fpath = uncropped_kwcoco_fpath.shrinkuser(home='$HOME')

        convert_options = []
        if collated:
            convert_options.append('--from-collated')
        if config['ignore_duplicates']:
            convert_options.append('--ignore-duplicates')
        convert_options_str = ' '.join(convert_options)

        queue.submit(ub.codeblock(
            rf'''
            [[ -f {uncropped_kwcoco_fpath} ]] || AWS_DEFAULT_PROFILE={aws_profile} python -m watch.cli.ta1_stac_to_kwcoco \
                "{uncropped_catalog_fpath}" \
                --outpath="{uncropped_kwcoco_fpath}" \
                {convert_options_str} \
                --jobs "min(avail,8)"
            '''))

        uncropped_coco_paths.append(uncropped_kwcoco_fpath)

    if len(uncropped_coco_paths) == 1:
        uncropped_final_kwcoco_fpath = uncropped_coco_paths[0]
    else:
        union_suffix = ub.hash_data([p.name for p in uncropped_coco_paths])[0:8]
        uncropped_final_kwcoco_fpath = uncropped_dpath / f'data_{union_suffix}.kwcoco.json'
        uncropped_final_kwcoco_fpath = uncropped_final_kwcoco_fpath.shrinkuser(home='$HOME')
        uncropped_multi_src_part = ' '.join(['"{}"'.format(p) for p in uncropped_coco_paths])
        queue.submit(ub.codeblock(
            rf'''
            # COMBINE Uncropped datasets
            [[ -f {uncropped_final_kwcoco_fpath} ]] || python -m kwcoco union \
                --src {uncropped_multi_src_part} \
                --dst "{uncropped_final_kwcoco_fpath}" \
                --workers="min(avail,max(all/2,8))" \
                --enable_video_stats=False \
                --overwrite=warp \
                --target_gsd=10
            '''))

    uncropped_prep_kwcoco_fpath = uncropped_dpath / 'data_prepped.kwcoco.json'
    uncropped_prep_kwcoco_fpath = uncropped_prep_kwcoco_fpath.shrinkuser(home='$HOME')

    select_images_query = config['select_images']
    if select_images_query:
        suffix = '_' + ub.hash_data(select_images_query)[0:8]
        # Debugging
        small_uncropped_kwcoco_fpath = uncropped_kwcoco_fpath.augment(suffix=suffix)
        queue.submit(ub.codeblock(
            rf'''
            # SUBSET Uncropped datasets (usually for debugging)
            python -m kwcoco subset \
                --src="{uncropped_final_kwcoco_fpath}" \
                --dst="{small_uncropped_kwcoco_fpath}" \
                --select_images='{select_images_query}'
            '''))

        uncropped_final_kwcoco_fpath = small_uncropped_kwcoco_fpath
        uncropped_prep_kwcoco_fpath = uncropped_prep_kwcoco_fpath.augment(suffix=suffix)
        aligned_kwcoco_fpath = aligned_kwcoco_fpath.augment(suffix=suffix)
        # --populate-watch-fields \

    queue.submit(ub.codeblock(
        rf'''
        # PREPARE Uncropped datasets (usually for debugging)
        [[ -f {uncropped_prep_kwcoco_fpath} ]] || AWS_DEFAULT_PROFILE={aws_profile} python -m watch.cli.coco_add_watch_fields \
            --src "{uncropped_final_kwcoco_fpath}" \
            --dst "{uncropped_prep_kwcoco_fpath}" \
            --workers="min(avail,max(all/2,8))" \
            --enable_video_stats=False \
            --overwrite=warp \
            --target_gsd=10
        '''))

    # region_model_str = ' '.join([shlex.quote(str(p)) for p in region_models])

    cache_crops = 1
    if cache_crops:
        align_keep = 'img'
    else:
        align_keep = 'none'

    # align_workers = '"min(avail,max(all/2,8))"'
    align_workers = config['align_workers']

    debug_valid_regions = config['debug']
    align_visualize = config['debug']

    # PROJ_DEBUG=3
    job_environ_str = ' '.join([
        # 'PROJ_DEBUG=3',
        f'AWS_DEFAULT_PROFILE={aws_profile}',
    ])
    if job_environ_str:
        job_environ_str += ' '
    queue.submit(ub.codeblock(
        rf'''
        # MAIN WORKHORSE CROP IMAGES
        # Crop big images to the geojson regions
        {job_environ_str} python -m watch.cli.coco_align_geotiffs \
            --src "{uncropped_prep_kwcoco_fpath}" \
            --dst "{aligned_kwcoco_fpath}" \
            --regions "{region_dpath / '*.geojson'}" \
            --workers={align_workers} \
            --context_factor=1 \
            --geo_preprop=auto \
            --keep={align_keep} \
            --visualize={align_visualize} \
            --debug_valid_regions={debug_valid_regions} \
            --rpc_align_method affine_warp
        '''))

    # TODO:
    # Project annotation from latest annotations subdir
    # Prepare splits
    # Add baseline datasets to DVC

    if config['visualize']:
        queue.submit(ub.codeblock(
            rf'''
            # Update to whatever the state of the annotations submodule is
            python -m watch visualize \
                --src "{aligned_kwcoco_fpath}" \
                --draw_anns=False \
                --draw_imgs=True \
                --channels="red|green|blue" \
                --animate=True --workers=auto
            '''))

    if 1:

        site_model_dpath = (dvc_dpath / 'annotations/site_models').shrinkuser(home='$HOME')
        region_model_dpath = (dvc_dpath / 'annotations/region_models').shrinkuser(home='$HOME')

        viz_part = '--viz_dpath=auto' if config['visualize'] else ''

        queue.submit(ub.codeblock(
            rf'''
            # Update to whatever the state of the annotations submodule is
            python -m watch project_annotations \
                --src "{aligned_kwcoco_fpath}" \
                --dst "{aligned_kwcoco_fpath}" \
                --site_models="{site_model_dpath}/*.geojson" \
                --region_models="{region_model_dpath}/*.geojson" {viz_part}
            '''))

    if config['visualize']:
        queue.submit(ub.codeblock(
            rf'''
            # Update to whatever the state of the annotations submodule is
            python -m watch visualize \
                --src "{aligned_kwcoco_fpath}" \
                --draw_anns=True \
                --draw_imgs=False \
                --channels="red|green|blue" \
                --animate=True --workers=auto
            '''))

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
        if config['serial']:
            queue.serial_run()
        else:
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

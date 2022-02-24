"""
An end-to-end script for calling all the scripts needed to

* Pulls the STAC catalog that points to processed large image tiles
* Creates a virtual Uncropped kwcoco dataset that points to the large image tiles
* Crops the dataset to create an aligned TA2 dataset


TODO:
    - [ ] Rename to schedule_ta2_dataset


Examples:

DATASET_SUFFIX=TA1_FULL_SEQ_KR_S001_CLOUD_LT_10
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/eval2/master_collation_working/KR_S001.unique.fixed_ls_ids.cloudcover_lt_10.output



DVC_DPATH=$(python -m watch.cli.find_dvc)
S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input
DATASET_SUFFIX=Drop2-TA1-2022-02-24
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath=$S3_FPATH \
    --dvc_dpath=$DVC_DPATH \
    --collated=True \
    --serial=True --run=0


python -m watch.cli.coco_add_watch_fields \
    --src $INPUT_COCO_FPATH \
    --dst $INPUT_COCO_FPATH.prepped \
    --workers 16 \
    --target_gsd=10

kwcoco subset \
    --src="$HOME/data/dvc-repos/smart_watch_dvc/Uncropped-Drop2-TA1-2022-02-24/data.kwcoco.json" \
    --dst="$HOME/data/dvc-repos/smart_watch_dvc/Uncropped-Drop2-TA1-2022-02-24/small_data.kwcoco.json" \
    --select_images '.id < 20'

AWS_DEFAULT_PROFILE=iarpa python -m watch.cli.coco_align_geotiffs \
    --src "$HOME/data/dvc-repos/smart_watch_dvc/Uncropped-Drop2-TA1-2022-02-24/small_data.kwcoco.json" \
    --dst "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-02-24/small_data.kwcoco.json" \
    --regions "$HOME/data/dvc-repos/smart_watch_dvc/annotations/region_models/*.geojson" \
    --workers="min(avail,max(all/2,8))" \
    --context_factor=1 \
    --geo_preprop=auto \
    --visualize False \
    --keep none \
    --rpc_align_method affine_warp

"""


import scriptconfig as scfg
import ubelt as ub


class PrepareTA2Config(scfg.Config):
    default = {
        'dataset_suffix': scfg.Value(None, help=''),
        's3_fpath': scfg.Value(None, help=''),
        'dvc_dpath': scfg.Value('auto', help=''),
        'run': scfg.Value('0', help=''),
        'collated': scfg.Value(True, help='set to false if the input data is not collated'),
        'serial': scfg.Value(False, help='if True use serial mode'),
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

    dvc_dpath = config['dvc_dpath']
    if dvc_dpath == 'auto':
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
    dvc_dpath = ub.Path(dvc_dpath)

    s3_fpath = config['s3_fpath']

    aligned_bundle_name = f'Aligned-{config["dataset_suffix"]}'
    uncropped_bundle_name = f'Uncropped-{config["dataset_suffix"]}'

    region_dpath = dvc_dpath / 'annotations/region_models'
    # region_models = list(region_dpath.glob('*.geojson'))

    uncropped_dpath = dvc_dpath / uncropped_bundle_name
    uncropped_query_dpath = uncropped_dpath / '_query/items'

    uncropped_query_fpath = uncropped_query_dpath / ub.Path(s3_fpath).name
    uncropped_kwcoco_fpath = uncropped_dpath / 'data.kwcoco.json'
    uncropped_prep_kwcoco_fpath = uncropped_dpath / 'data_prepped.kwcoco.json'

    uncropped_ingress_dpath = uncropped_dpath / 'ingress'
    uncropped_catalog_fpath = uncropped_ingress_dpath / 'catalog.json'

    aligned_kwcoco_bundle = dvc_dpath / aligned_bundle_name
    aligned_kwcoco_fpath = aligned_kwcoco_bundle / 'data.kwcoco.json'

    region_dpath = region_dpath.shrinkuser(home='$HOME')
    uncropped_dpath = uncropped_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_dpath = uncropped_query_dpath.shrinkuser(home='$HOME')
    uncropped_query_fpath = uncropped_query_fpath.shrinkuser(home='$HOME')
    uncropped_kwcoco_fpath = uncropped_kwcoco_fpath.shrinkuser(home='$HOME')
    uncropped_prep_kwcoco_fpath = uncropped_prep_kwcoco_fpath.shrinkuser(home='$HOME')
    uncropped_ingress_dpath = uncropped_ingress_dpath.shrinkuser(home='$HOME')
    uncropped_catalog_fpath = uncropped_catalog_fpath.shrinkuser(home='$HOME')
    aligned_kwcoco_bundle = aligned_kwcoco_bundle.shrinkuser(home='$HOME')
    aligned_kwcoco_fpath = aligned_kwcoco_fpath.shrinkuser(home='$HOME')

    # queue = tmux_queue.SerialQueue()
    queue = tmux_queue.TMUXMultiQueue(name='teamfeat', size=1, gres=None)

    aws_profile = 'iarpa'

    queue.submit(ub.codeblock(
        f'''
        mkdir -p {uncropped_query_dpath}
        [[ -f {uncropped_query_fpath} ]] || aws s3 --profile {aws_profile} cp "{s3_fpath}" "{uncropped_query_dpath}"
        '''))

    queue.submit(ub.codeblock(
        rf'''
        [[ -f {uncropped_catalog_fpath} ]] || python -m watch.cli.baseline_framework_ingress \
            --aws_profile {aws_profile} \
            --jobs avail \
            --virtual \
            --outdir "{uncropped_ingress_dpath}" \
            "{uncropped_query_fpath}"
        '''))

    if config['collated']:
        collated_str = '--from-collated'
    else:
        collated_str = ''

    queue.submit(ub.codeblock(
        rf'''
        [[ -f {uncropped_kwcoco_fpath} ]] || AWS_DEFAULT_PROFILE={aws_profile} python -m watch.cli.ta1_stac_to_kwcoco \
            "{uncropped_catalog_fpath}" \
            --outpath="{uncropped_kwcoco_fpath}" \
            --populate-watch-fields \
            {collated_str} \
            --jobs "min(avail,8)"
        '''))

    queue.submit(ub.codeblock(
        rf'''
        [[ -f {uncropped_prep_kwcoco_fpath} ]] || AWS_DEFAULT_PROFILE={aws_profile} python -m watch.cli.coco_add_watch_fields \
            --src "{uncropped_kwcoco_fpath}" \
            --dst "{uncropped_prep_kwcoco_fpath}" \
            --workers="min(avail,max(all/2,8))" \
            --enable_video_stats=False \
            --overwrite=warp \
            --target_gsd=10
        '''))

    # region_model_str = ' '.join([shlex.quote(str(p)) for p in region_models])

    queue.submit(ub.codeblock(
        rf'''
        AWS_DEFAULT_PROFILE={aws_profile} python -m watch.cli.coco_align_geotiffs \
            --src "{uncropped_prep_kwcoco_fpath}" \
            --dst "{aligned_kwcoco_fpath}" \
            --regions "{region_dpath / '*.geojson'}" \
            --workers="min(avail,max(all/2,8))" \
            --context_factor=1 \
            --geo_preprop=auto \
            --visualize False \
            --keep none \
            --rpc_align_method affine_warp
        '''))

    # TODO:
    # Project annotation from latest annotations subdir
    # Prepare splits
    # Add baseline datasets to DVC

    '''
    # Update to whatever the state of the annotations submodule is
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    python -m watch project_annotations \
        --src $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
        --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
        --site_models="$DVC_DPATH/annotations/site_models/*.geojson"

    DVC_DPATH=$(python -m watch.cli.find_dvc)
    python -m watch.cli.prepare_splits \
        --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
        --run=1 --serial=True

    dvc add Drop2-Aligned-TA1-2022-02-15/data_*.kwcoco.json



    '''

    queue.rprint()

    if config['run']:
        agg_state = None
        if config['serial']:
            queue.serial_run()
        else:
            queue.run()
        if config['follow']:
            agg_state = queue.monitor()
        if not config['keep_sessions']:
            if agg_state is not None and not agg_state['errored']:
                queue.kill()


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

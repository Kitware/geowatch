#!/bin/bash
__doc__="
The script that builds the drop4 dataset and contains debugging information

See Also:
    ~/code/watch/watch/cli/prepare_ta2_dataset.py
"


## Create a demo region file
#xdoctest watch.demo.demo_region demo_khq_region_fpath
#ROOT_DPATH="$DATA_DVC_DPATH"



# Requires alternative-enhancement transcrypt branch
#transcrypt -p "$WATCH_TRANSCRYPT_SECRET"

# TODO: add a resource for this?
source "$HOME"/code/watch/secrets/secrets

DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
SENSORS=TA1-S2-L8-ACC
#SENSORS=TA1-S2-ACC
#SENSORS=L2-S2-L8
DATASET_SUFFIX=Drop4-2022-07-25-c30-$SENSORS
REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

#DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
#REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/NZ_R001.*"
#SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=30 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DATA_DVC_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --requester_pays=False \
    --fields_workers=20 \
    --convert_workers=8 \
    --max_queue_size=12 \
    --align_workers=12 \
    --cache=0 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --include_channels="blue|green|red|nir|swir16|swir22" \
    --visualize=0 \
    --target_gsd=30 \
    --backend=tmux --run=1



build_drop4_BAS(){
    source "$HOME"/code/watch/secrets/secrets
    SENSORS=TA1-S2-L8-ACC
    DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_SUFFIX=Drop4-2022-07-28-c20-$SENSORS
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=20 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated True \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --requester_pays=False \
        --fields_workers=20 \
        --convert_workers=8 \
        --max_queue_size=12 \
        --align_workers=12 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --include_channels="blue|green|red|nir|swir16|swir22|cloudmask" \
        --visualize=1 \
        --target_gsd=30 \
        --force_nodata=-9999 \
        --cache="before:aligned_kwcoco" \
        --align_keep=none \
        --backend=tmux --run=1
}


build_drop4_v2_BAS(){
    source "$HOME"/code/watch/secrets/secrets
    SENSORS=TA1-S2-L8-ACC
    DATA_DVC_DPATH=$HOME/data/dvc-repos/smart_data_dvc
    #DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")

    DATASET_SUFFIX=Drop4-2022-08-08-$SENSORS
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=40 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated True \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --requester_pays=False \
        --fields_workers=20 \
        --convert_workers=8 \
        --max_queue_size=12 \
        --align_workers=12 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --visualize=1 \
        --target_gsd=30 \
        --force_nodata=-9999 \
        --cache=0 \
        --align_keep=none \
        --backend=tmux --run=1
}


build_drop4_v2_SC(){
    source "$HOME"/code/watch/secrets/secrets
    SENSORS=TA1-S2-WV-PD-ACC
    DATA_DVC_DPATH=$HOME/data/dvc-repos/smart_data_dvc
    #DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")

    DATASET_SUFFIX=Drop4-2022-08-08-$SENSORS
    REGION_GLOBSTR="$DATA_DVC_DPATH/subregions/*.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=40 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated True \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --requester_pays=False \
        --fields_workers=20 \
        --convert_workers=8 \
        --max_queue_size=12 \
        --align_workers=12 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --visualize=1 \
        --visualize_only_boxes=False \
        --target_gsd=4 \
        --force_nodata=-9999 \
        --cache=1 \
        --align_keep=img \
        --backend=tmux --run=1

}


dvc_add_SC(){
    __hack__="
    import kwcoco
    dset = kwcoco.CocoDataset('data.kwcoco.json')

    ignore_regions = {
        'CN_C001',
    }
    vali_regions = {
        'KR_R001',
        'KR_R002',
        'US_R007',
    }

    videos = dset.videos()
    is_vali_flag = [any(n.startswith(r) for r in vali_regions) and not any(n.startswith(r) for r in ignore_regions) for n in videos.lookup('name')]
    is_train_flag = [not any(n.startswith(r) for r in vali_regions) and not any(n.startswith(r) for r in ignore_regions) for n in videos.lookup('name')]

    vali_gids = list(ub.flatten(videos.compress(is_vali_flag).images))
    train_gids = list(ub.flatten(videos.compress(is_train_flag).images))

    vali_dset = dset.subset(vali_gids)
    train_dset = dset.subset(train_gids)

    vali_dset.fpath = 'data_vali.kwcoco.json'
    vali_dset.dump(vali_dset.fpath)

    train_dset.fpath = 'data_train.kwcoco.json'
    train_dset.dump(train_dset.fpath)
    "

    #python -m kwcoco subset \
    #--src imgonly-BR_R004_0020_box.kwcoco.json \
    #--dst foo.kwcoco.json \
    #--select_videos '(.name | startswith("KR_R002")) or (.name | startswith("KR_R001")) or (.name | startswith("US_R007"))'

    DATA_DVC_DPATH=$HOME/data/dvc-repos/smart_data_dvc
    cd "$DATA_DVC_DPATH"/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC

    dvc unprotect -- */L8 */S2 *.zip viz512_anns

    python -m watch.cli.prepare_splits data.kwcoco.json --cache=0 --run=1

    __hack__="
    import shutil
    d = ub.Path('viz512_anns')
    paths = list(ub.Path('_viz512').glob('*/_anns/red_green_blue'))
    for p in ub.ProgIter(paths):
        dst_dpath = d / p.parent.parent.name
        shutil.copytree(p, dst_dpath)
    "
    mkdir -p viz512_anns
    cp _viz512/*/*ann*.gif ./viz512_anns
    cp _viz512/*/*ann*.gif ./viz512_anns

    7z a splits.zip data*.kwcoco.json

    # Cd into the bundle we want to add
    ls -- */L8
    ls -- */S2
    ls -- */*.json

    dvc add -- */PD */WV */S2 viz512_anns && dvc push -r aws -R . && git commit -am "Add Drop4 SC Images" && git push  && \
    dvc add -- *.zip && dvc push -r aws -R . && git commit -am "Add Drop4 SC Annots" && git push 
    #dvc add -- */L8 */S2 && dvc push -r aws -R . && git commit -am "Add Drop4" && git push 

    #dvc add data_*nowv*.kwcoco.json
}




#mkdir -p "$DEMO_DPATH"
## Create the search json wrt the sensors and processing level we want
#python -m watch.cli.stac_search_build \
#    --start_date="$START_DATE" \
#    --end_date="$END_DATE" \
#    --cloud_cover=40 \
#    --sensors=L2 \
#    --out_fpath "$SEARCH_FPATH"
#cat "$SEARCH_FPATH"

## Delete this to prevent duplicates
#rm -f "$RESULT_FPATH"
## Create the .input file
#python -m watch.cli.stac_search \
#    --region_file "$REGION_FPATH" \
#    --search_json "$SEARCH_FPATH" \
#    --mode area \
#    --verbose 2 \
#    --outfile "${RESULT_FPATH}"

#SEARCH_FPATH=$DEMO_DPATH/stac_search.json
#RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input
#START_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.start_date' "$REGION_FPATH")
#END_DATE=$(jq -r '.features[] | select(.properties.type=="region") | .properties.end_date' "$REGION_FPATH")
#REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
#PREPARE_DPATH=$ROOT_DPATH/_prepare/"$DATASET_SUFFIX"


rgb_medium_drop4_only(){
    source "$HOME"/code/watch/secrets/secrets

    DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    SENSORS=TA1-S2-L8-ACC
    #SENSORS=TA1-S2-ACC
    #SENSORS=L2-S2-L8
    DATASET_SUFFIX=Drop4-2022-07-24-c10-rgb-$SENSORS
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=10 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated True \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --requester_pays=False \
        --max_products_per_region=100 \
        --fields_workers=20 \
        --convert_workers=8 \
        --max_queue_size=12 \
        --align_workers=12 \
        --cache=1 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --include_channels="blue|green|red" \
        --visualize=True \
        --backend=serial --run=1
}


small_onesite(){
    DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    source ~/code/watch/secrets/secrets
    SENSORS=TA1-S2-L8-WV-PD-ACC
    #SENSORS=L2-S2
    #SENSORS=L2-L8
    #SENSORS=TA1-S2-ACC
    #SENSORS=TA1-L8-ACC
    #SENSORS=L2-S2-L8
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/BR_R001.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/BR_R001*.geojson"
    DATASET_SUFFIX=Drop4-2022-07-28-$SENSORS-onesite

    # Test credentials
    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=10 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated True \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --requester_pays=False \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --max_products_per_region=3 \
        --fields_workers=10 \
        --convert_workers=10 \
        --align_workers=10 \
        --cache=0 \
        --include_channels="red|blue|green" \
        --ignore_duplicates=1 \
        --visualize=1 \
        --backend=serial --run=1
        #--fields_workers=100 \
        #--convert_workers=8 \
        #--align_workers=26 \
}

small_teregions(){
    DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    SENSORS=L2-S2
    DATASET_SUFFIX=Drop4-2022-07-18-$SENSORS-small-teregion
    #REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*_R*.geojson"
    #SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*_R*.geojson"
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/BR_R001.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/BR_R001_*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=1 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --max_products_per_region=8 \
        --fields_workers=0 \
        --convert_workers=0 \
        --align_workers=0 \
        --cache=0 \
        --ignore_duplicates=1 \
        --target_gsd=30 \
        --visualize=True \
        --backend=serial --run=0
}

small_allsites(){
    DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_SUFFIX=Drop4-S2-L2A-2022-07-15-demo8
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=1 \
        --sensors=L2 \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --max_products_per_region=8 \
        --max_regions=None \
        --fields_workers=4 \
        --convert_workers=4 \
        --align_workers=4 \
        --cache=1 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --target_gsd=30 \
        --visualize=True \
        --backend=tmux --run=1


    DATA_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_SUFFIX=Drop4-S2-L2A-2022-07-16-c40
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"
    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"
    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=40 \
        --sensors=L2 \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --max_products_per_region=None \
        --max_regions=None \
        --fields_workers=100 \
        --convert_workers=4 \
        --align_workers=16 \
        --cache=1 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --target_gsd=30 \
        --visualize=True \
        --backend=tmux --run=1
}


_Debugging(){
    AWS_DEFAULT_PROFILE=iarpa AWS_REQUEST_PAYER=requester python -m watch.cli.coco_align_geotiffs \
        --src /home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc-hdd/Uncropped-Drop4-2022-07-18-c10-TA1-S2-ACC/data_BR_R005_fielded.kwcoco.json \
        --dst /home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop4-2022-07-18-c10-TA1-S2-ACC/imgonly-BR_R005.kwcoco.json \
        --regions /home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc-hdd/annotations/region_models/BR_R005.geojson \
        --context_factor=1 --geo_preprop=auto --keep=roi-img \
        --include_channels="red|green|blue" \
        --visualize=False --debug_valid_regions=False \
        --rpc_align_method affine_warp --verbose=10 --aux_workers=0 --workers=20

        #--exclude_channels="tci:3|B05|B06|B07|B8A|B09" \
}


dvc_add(){
    cd Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC

    dvc unprotect -- */L8 */S2 *.zip viz512_anns

    python -m watch.cli.prepare_splits data.kwcoco.json --cache=0 --run=1
    #--backend=serial

    mkdir -p viz512_anns
    cp _viz512/*/*ann*.gif ./viz512_anns

    7z a splits.zip data*.kwcoco.json

    # Cd into the bundle we want to add
    ls -- */L8
    ls -- */S2
    ls -- */*.json

    dvc add -- */L8 */S2 *.zip viz512_anns && dvc push -r aws -R . && git commit -am "Add Drop4" && git push 
    dvc add -- */L8 */S2 && dvc push -r aws -R . && git commit -am "Add Drop4" && git push 

    #dvc add data_*nowv*.kwcoco.json
}


update_local(){
    DATA_DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
    cd "$DATA_DVC_DPATH"/Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC
    git pull
    dvc pull -r aws splits.zip.dvc
    7z x splits.zip -y

}



prepare_qfabric(){
    source "$HOME"/code/watch/secrets/secrets

    DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")/public
    SENSORS=worldview-nitf
    DATASET_SUFFIX=QFabric-c30-$SENSORS
    REGION_GLOBSTR="$DATA_DVC_DPATH/annotations-qfabric/orig/region_models/*.geojson"
    SITE_GLOBSTR="$DATA_DVC_DPATH/annotations-qfabric/orig/site_models/*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=30 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated True \
        --dvc_dpath="$DATA_DVC_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --requester_pays=False \
        --fields_workers=20 \
        --convert_workers=8 \
        --max_queue_size=12 \
        --align_workers=12 \
        --cache=0 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --visualize=0 \
        --target_gsd=2 \
        --backend=tmux --run=0
        
}

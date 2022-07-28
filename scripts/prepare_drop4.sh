#!/bin/bash
__doc__="
The script that builds the drop4 dataset and contains debugging information

See Also:
    ~/code/watch/watch/cli/prepare_ta2_dataset.py
"


## Create a demo region file
#xdoctest watch.demo.demo_region demo_khq_region_fpath
#ROOT_DPATH="$DVC_DPATH"



# Requires alternative-enhancement transcrypt branch
#transcrypt -p "$WATCH_TRANSCRYPT_SECRET"

# TODO: add a resource for this?
source "$HOME"/code/watch/secrets/secrets

DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
SENSORS=TA1-S2-L8-ACC
#SENSORS=TA1-S2-ACC
#SENSORS=L2-S2-L8
DATASET_SUFFIX=Drop4-2022-07-25-c30-$SENSORS
REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/*.geojson"
SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

#DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
#REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/NZ_R001.*"
#SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=30 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated False \
    --dvc_dpath="$DVC_DPATH" \
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



build_drop4_all_sensors(){
    source "$HOME"/code/watch/secrets/secrets
    SENSORS=TA1-S2-L8-WV-PL-ACC
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_SUFFIX=Drop4-2022-07-26-c40-$SENSORS
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=40 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DVC_DPATH" \
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
        --visualize=1 \
        --target_gsd=30 \
        --backend=tmux --run=1
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

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    SENSORS=TA1-S2-L8-ACC
    #SENSORS=TA1-S2-ACC
    #SENSORS=L2-S2-L8
    DATASET_SUFFIX=Drop4-2022-07-24-c10-rgb-$SENSORS
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=10 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DVC_DPATH" \
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
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    source ~/code/watch/secrets/secrets
    SENSORS=TA1-S2-L8-WV-PL-ACC
    #SENSORS=L2-S2
    #SENSORS=L2-L8
    #SENSORS=TA1-S2-ACC
    #SENSORS=TA1-L8-ACC
    #SENSORS=L2-S2-L8
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/BR_R001.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/BR_R001*.geojson"
    DATASET_SUFFIX=Drop4-2022-07-28-$SENSORS-onesite

    # Test credentials
    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=10 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DVC_DPATH" \
        --aws_profile=iarpa \
        --requester_pays=False \
        --region_globstr="$REGION_GLOBSTR" \
        --site_globstr="$SITE_GLOBSTR" \
        --max_products_per_region=10 \
        --fields_workers=30 \
        --convert_workers=20 \
        --align_workers=20 \
        --cache=1 \
        --include_channels="blue|green|red" \
        --ignore_duplicates=1 \
        --visualize=True \
        --backend=serial --run=1
        #--fields_workers=100 \
        #--convert_workers=8 \
        #--align_workers=26 \
}

small_teregions(){
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    SENSORS=L2-S2
    DATASET_SUFFIX=Drop4-2022-07-18-$SENSORS-small-teregion
    #REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/*_R*.geojson"
    #SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*_R*.geojson"
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/BR_R001.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/BR_R001_*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=1 \
        --sensors="$SENSORS" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DVC_DPATH" \
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
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_SUFFIX=Drop4-S2-L2A-2022-07-15-demo8
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=1 \
        --sensors=L2 \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DVC_DPATH" \
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


    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_SUFFIX=Drop4-S2-L2A-2022-07-16-c40
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"
    #DATASET_SUFFIX=Test-Drop4-L2-2022-07-06
    #REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/NZ_R001.*"
    #SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"
    # Construct the TA2-ready dataset
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --stac_query_mode=auto \
        --cloud_cover=40 \
        --sensors=L2 \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --dvc_dpath="$DVC_DPATH" \
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

    python -m watch.cli.prepare_splits data.kwcoco.json --run=0

    python -m watch.cli.prepare_splits \
        --base_fpath=data.kwcoco.json \
        --run=0 --backend=serial
    7z a splits.zip data*.kwcoco.json

    cd Aligned-Drop4-2022-07-25-c30-TA1-S2-L8-ACC

    # Cd into the bundle we want to add
    ls -- */L8
    ls -- */S2
    ls -- */*.json

    dvc add -- */L8 */S2 
    dvc add -- *.zip
    git commit -am "Add Drop4"

    dvc push -r aws -R .

    #dvc add data_*nowv*.kwcoco.json
    
}

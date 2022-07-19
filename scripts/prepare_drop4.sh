#!/bin/bash
__doc__="
The script that builds the drop4 dataset and contains debugging information

See Also:
    ~/code/watch/watch/cli/prepare_ta2_dataset.py
"


## Create a demo region file
#xdoctest watch.demo.demo_region demo_khq_region_fpath
#ROOT_DPATH="$DVC_DPATH"

DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
DATASET_SUFFIX=Drop4-L2-2022-07-10
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
    --sensors=L2 \
    --api_key=env:SMART_STAC_API_KEY \
    --collated False \
    --dvc_dpath="$DVC_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --fields_workers=8 \
    --convert_workers=8 \
    --max_queue_size=100 \
    --align_workers=26 \
    --cache=0 \
    --ignore_duplicates=1 \
    --visualize=True \
    --serial=True --run=0

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

small_onesite(){
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    #SENSORS=TA1-S2-L8-ACC
    #SENSORS=L2-S2
    SENSORS=L2-L8
    #SENSORS=L2-S2-L8
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/AE_R001.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/AE_R001.geojson"
    DATASET_SUFFIX=Drop4-2022-07-18-$SENSORS-demo


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
        --fields_workers=100 \
        --convert_workers=8 \
        --max_products_per_region=3 \
        --align_workers=26 \
        --cache=1 \
        --ignore_duplicates=1 \
        --visualize=True \
        --backend=serial --run=0
}

small_allsites(){
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_SUFFIX=Drop4-L2-2022-07-14-demo3
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
        --fields_workers=26 \
        --convert_workers=8 \
        --align_workers=26 \
        --cache=0 \
        --ignore_duplicates=1 \
        --target_gsd=30 \
        --visualize=True \
        --backend=serial --run=1
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
